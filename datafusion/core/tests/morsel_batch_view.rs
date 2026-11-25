// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! End-to-end SQL parity and invariant tests for in-pipeline `BatchView`
//! selection vectors across Filter -> HashJoin, Filter -> GROUP BY / COUNT(*),
//! Filter -> Projection, stacked filters, sliced batches, NULL predicates,
//! zero-column batches, and memory-limited runs (#169).

#![cfg(feature = "sql")]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrow::array::{
    Array, DictionaryArray, Float64Array, Int32Array, Int64Array, RecordBatch,
    StringArray,
};
use arrow::datatypes::{DataType, Field, Int32Type, Schema};
use arrow::util::display::array_value_to_string;
use datafusion::common::config::MorselMode;
use datafusion::datasource::MemTable;
use datafusion::error::Result;
use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryPool};
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::logical_expr::{ColumnarValue, Volatility, create_udf};
use datafusion::physical_plan::{collect, displayable};
use datafusion::prelude::{SessionConfig, SessionContext};

fn session(mode: MorselMode, workers: usize, batch_size: usize) -> SessionContext {
    let mut config = SessionConfig::new()
        .with_target_partitions(workers)
        .with_batch_size(batch_size);
    config.options_mut().execution.morsel_mode = mode;
    SessionContext::new_with_config(config)
}

fn sorted_rows(batches: &[RecordBatch]) -> Result<Vec<Vec<String>>> {
    let mut rows = Vec::new();
    for batch in batches {
        for r in 0..batch.num_rows() {
            let mut row = Vec::with_capacity(batch.num_columns());
            for c in 0..batch.num_columns() {
                let col = batch.column(c);
                if col.is_null(r) {
                    row.push("<NULL>".to_owned());
                } else {
                    row.push(array_value_to_string(col.as_ref(), r)?);
                }
            }
            rows.push(row);
        }
    }
    rows.sort();
    Ok(rows)
}

/// Build a MemTable whose batches are Arrow slices with non-zero offsets,
/// NULLs in keys/values/predicates, and wide payload columns.
fn sliced_fact_table() -> Result<Arc<MemTable>> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("k", DataType::Int64, true),
        Field::new("sk", DataType::Utf8, true),
        Field::new("grp", DataType::Int32, true),
        Field::new("val", DataType::Int64, true),
        Field::new("fval", DataType::Float64, true),
        Field::new("gate", DataType::Int32, true),
        Field::new("payload", DataType::Utf8, true),
    ]));

    let total = 240usize;
    let ids: Vec<i64> = (0..total as i64).collect();
    let ks: Vec<Option<i64>> = (0..total)
        .map(|i| {
            if i % 11 == 0 {
                None
            } else {
                Some((i % 25) as i64)
            }
        })
        .collect();
    let sks: Vec<Option<String>> = (0..total)
        .map(|i| {
            if i % 13 == 0 {
                None
            } else {
                Some(format!("key-{:02}", i % 20))
            }
        })
        .collect();
    let grps: Vec<Option<i32>> = (0..total)
        .map(|i| {
            if i % 9 == 0 {
                None
            } else {
                Some((i % 7) as i32)
            }
        })
        .collect();
    let vals: Vec<Option<i64>> = (0..total)
        .map(|i| {
            if i % 10 == 0 {
                None
            } else {
                Some((i as i64) * 3 - 50)
            }
        })
        .collect();
    let fvals: Vec<Option<f64>> = (0..total)
        .map(|i| {
            if i % 8 == 0 {
                None
            } else {
                Some((i as f64) * 0.5)
            }
        })
        .collect();
    let gates: Vec<Option<i32>> = (0..total)
        .map(|i| {
            if i % 6 == 0 {
                None
            } else {
                Some((i % 10) as i32)
            }
        })
        .collect();
    let payloads: Vec<Option<String>> = (0..total)
        .map(|i| {
            if i % 15 == 0 {
                None
            } else {
                Some(format!("payload-{i:04}-{}", "p".repeat(16)))
            }
        })
        .collect();

    let big = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(ids)),
            Arc::new(Int64Array::from(ks)),
            Arc::new(StringArray::from(sks)),
            Arc::new(Int32Array::from(grps)),
            Arc::new(Int64Array::from(vals)),
            Arc::new(Float64Array::from(fvals)),
            Arc::new(Int32Array::from(gates)),
            Arc::new(StringArray::from(payloads)),
        ],
    )?;

    // Slice with non-zero offsets and uneven sizes, including an empty batch.
    let part0 = vec![
        big.slice(5, 37),
        RecordBatch::new_empty(Arc::clone(&schema)),
        big.slice(42, 41),
    ];
    let part1 = vec![big.slice(83, 53), big.slice(136, 49)];
    let part2 = vec![big.slice(185, 50)];
    Ok(Arc::new(MemTable::try_new(
        schema,
        vec![part0, part1, part2],
    )?))
}

fn sliced_dim_table() -> Result<Arc<MemTable>> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("dk", DataType::Int64, true),
        Field::new("dsk", DataType::Utf8, true),
        Field::new("dgrp", DataType::Int32, true),
        Field::new("weight", DataType::Int64, true),
        Field::new("dflag", DataType::Int32, true),
    ]));
    // Include duplicate keys for fanout and NULL keys.
    let mut dks = Vec::new();
    let mut dsks = Vec::new();
    let mut dgrps = Vec::new();
    let mut weights = Vec::new();
    let mut dflags = Vec::new();
    for i in 0..40usize {
        let key = if i % 10 == 0 {
            None
        } else {
            Some((i % 15) as i64)
        };
        dks.push(key);
        dsks.push(key.map(|k| format!("key-{k:02}")));
        dgrps.push(if i % 7 == 0 {
            None
        } else {
            Some((i % 5) as i32)
        });
        weights.push(Some((i as i64) * 10));
        dflags.push(if i % 4 == 0 {
            None
        } else {
            Some((i % 3) as i32)
        });
    }
    let big = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(dks)),
            Arc::new(StringArray::from(dsks)),
            Arc::new(Int32Array::from(dgrps)),
            Arc::new(Int64Array::from(weights)),
            Arc::new(Int32Array::from(dflags)),
        ],
    )?;
    Ok(Arc::new(MemTable::try_new(
        schema,
        vec![vec![big.slice(2, 18)], vec![big.slice(20, 18)]],
    )?))
}

async fn assert_sql_parity(
    fact: Arc<MemTable>,
    dim: Arc<MemTable>,
    sql: &str,
    workers: usize,
    batch_size: usize,
) -> Result<()> {
    let stock_ctx = session(MorselMode::Off, workers, batch_size);
    let morsel_ctx = session(MorselMode::Required, workers, batch_size);
    stock_ctx.register_table("fact", fact.clone())?;
    stock_ctx.register_table("dim", dim.clone())?;
    morsel_ctx.register_table("fact", fact)?;
    morsel_ctx.register_table("dim", dim)?;

    let stock_plan = stock_ctx.sql(sql).await?.create_physical_plan().await?;
    let morsel_plan = morsel_ctx.sql(sql).await?.create_physical_plan().await?;
    let display = displayable(morsel_plan.as_ref()).indent(false).to_string();
    assert_eq!(morsel_plan.name(), "MorselExec", "{sql}: {display}");
    assert_eq!(morsel_plan.schema(), stock_plan.schema(), "{sql}");

    let expected = collect(stock_plan, stock_ctx.task_ctx()).await?;
    let actual = collect(morsel_plan.clone(), morsel_ctx.task_ctx()).await?;
    for batch in &actual {
        assert_eq!(batch.schema(), morsel_plan.schema(), "{sql}");
    }
    assert_eq!(sorted_rows(&actual)?, sorted_rows(&expected)?, "{sql}");
    Ok(())
}

#[tokio::test]
async fn sql_parity_filtered_join_group_by_projection_and_sliced_batches() -> Result<()> {
    let fact = sliced_fact_table()?;
    let dim = sliced_dim_table()?;

    let queries = [
        // 1. Filter -> HashJoin probe (integer key, selective, NULL predicates, fanout)
        "SELECT f.id, f.payload, d.weight FROM fact f JOIN dim d ON f.k = d.dk WHERE f.gate > 5",
        // Filtered on both build and probe sides + post-join filter
        "SELECT f.id, d.weight + f.val AS total FROM fact f JOIN dim d ON f.k = d.dk WHERE f.gate >= 3 AND d.dflag > 0 AND d.weight > 50",
        // Filtered HashJoin with string key (CompactedKeys path)
        "SELECT f.id, f.sk, d.weight FROM fact f JOIN dim d ON f.sk = d.dsk WHERE f.gate = 2 OR f.gate = 7",
        // Filtered HashJoin with composite key (Rows + CompactedKeys path)
        "SELECT f.id, d.weight FROM fact f JOIN dim d ON f.k = d.dk AND f.sk = d.dsk WHERE f.gate > 4",
        // Filtered HashJoin with expression key
        "SELECT f.id, d.weight FROM fact f JOIN dim d ON f.k + 1 = d.dk + 1 WHERE f.gate > 6",
        // Filtered HashJoin -> COUNT(*) (zero-column output projection)
        "SELECT COUNT(*) AS n FROM fact f JOIN dim d ON f.k = d.dk WHERE f.gate > 5",
        // Filtered HashJoin -> GROUP BY (DirectGroupedState & GroupCountSink)
        "SELECT d.dgrp, COUNT(*) AS n, SUM(f.val) AS s, AVG(f.fval) AS a, MIN(f.val) AS mn, MAX(f.val) AS mx FROM fact f JOIN dim d ON f.k = d.dk WHERE f.gate > 4 GROUP BY d.dgrp",
        "SELECT d.dgrp, COUNT(*) AS n FROM fact f JOIN dim d ON f.k = d.dk WHERE f.gate > 4 GROUP BY d.dgrp",
        // 2. Filter -> GROUP BY / COUNT(*)
        "SELECT COUNT(*) AS n FROM fact WHERE gate > 5",
        "SELECT COUNT(*) AS n FROM fact WHERE gate > 100",
        "SELECT COUNT(*) AS n FROM fact WHERE gate IS NOT NULL",
        "SELECT grp, COUNT(*) AS n FROM fact WHERE gate > 4 GROUP BY grp",
        "SELECT grp, COUNT(*) AS n, COUNT(val) AS cv, SUM(val) AS sv, MIN(val) AS mn, MAX(val) AS mx, AVG(fval) AS af FROM fact WHERE gate > 3 GROUP BY grp",
        "SELECT grp, COUNT(*) AS n, SUM(val) AS sv FROM fact WHERE gate > 100 GROUP BY grp",
        // General GroupedState (string key + expression arg)
        "SELECT sk, COUNT(*) AS n, SUM(val + 1) AS sv FROM fact WHERE gate > 4 GROUP BY sk",
        // Multi-key GROUP BY
        "SELECT grp, sk, COUNT(*) AS n, SUM(val) AS sv FROM fact WHERE gate >= 2 GROUP BY grp, sk",
        // Ungrouped aggregate with filter
        "SELECT COUNT(*) AS n, SUM(val) AS sv, AVG(fval) AS af FROM fact WHERE gate > 5",
        // 3. Filter -> Projection (pure column, mixed expression, constant/zero-column, unfiltered controls)
        "SELECT id, payload FROM fact WHERE gate > 6",
        "SELECT id, payload, val * 2 + 1 AS v2 FROM fact WHERE gate > 6",
        "SELECT 42 AS c FROM fact WHERE gate > 6",
        "SELECT id, val + 1 AS v1 FROM fact",
        "SELECT id, payload FROM fact",
        // All-filtered, all-selected, and all-NULL runtime predicate controls
        "SELECT id, payload FROM fact WHERE gate > 100",
        "SELECT id, payload FROM fact WHERE id >= 0",
        "SELECT id, payload FROM fact WHERE gate = NULLIF(gate, gate)",
    ];

    for (workers, batch_size) in [(1, 13), (4, 17)] {
        for sql in queries {
            assert_sql_parity(
                Arc::clone(&fact),
                Arc::clone(&dim),
                sql,
                workers,
                batch_size,
            )
            .await?;
        }
    }
    Ok(())
}

#[tokio::test]
async fn projection_udf_never_evaluates_rejected_rows_in_filtered_pipeline() -> Result<()>
{
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("gate", DataType::Int64, false),
        Field::new("payload", DataType::Utf8, false),
    ]));
    let rows = 100usize;
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from_iter_values(0..rows as i64)),
            Arc::new(Int64Array::from_iter_values(
                (0..rows).map(|i| (i % 10) as i64),
            )),
            Arc::new(StringArray::from(
                (0..rows).map(|i| format!("p-{i}")).collect::<Vec<_>>(),
            )),
        ],
    )?;
    let table = Arc::new(MemTable::try_new(schema, vec![vec![batch]])?);

    for workers in [1usize, 4usize] {
        let evaluated_rows = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&evaluated_rows);
        let strict_udf = create_udf(
            "strict_double",
            vec![DataType::Int64],
            DataType::Int64,
            Volatility::Immutable,
            Arc::new(move |args| {
                let ColumnarValue::Array(arr) = &args[0] else {
                    unreachable!();
                };
                let ints = arr.as_any().downcast_ref::<Int64Array>().unwrap();
                counter.fetch_add(ints.len(), Ordering::Relaxed);
                for &v in ints.values() {
                    if v % 10 != 7 {
                        return datafusion::common::exec_err!(
                            "strict_double evaluated on rejected row id={v}"
                        );
                    }
                }
                let out: Int64Array = ints.values().iter().map(|&v| v * 2).collect();
                Ok(ColumnarValue::Array(Arc::new(out)))
            }),
        );

        let ctx = session(MorselMode::Required, workers, 32);
        ctx.register_table("t", table.clone())?;
        ctx.register_udf(strict_udf);

        // Only 10 rows have gate = 7 (ids 7, 17, ..., 97).
        let sql = "SELECT payload, strict_double(id) AS d FROM t WHERE gate = 7";
        let plan = ctx.sql(sql).await?.create_physical_plan().await?;
        assert_eq!(plan.name(), "MorselExec");
        let batches = collect(plan, ctx.task_ctx()).await?;
        let total: usize = batches.iter().map(RecordBatch::num_rows).sum();
        assert_eq!(total, 10);
        assert_eq!(
            evaluated_rows.load(Ordering::Relaxed),
            10,
            "workers={workers}: UDF must only run on the 10 selected rows"
        );
    }
    Ok(())
}

#[tokio::test]
async fn filtered_hash_join_build_under_tight_memory_pool_parity() -> Result<()> {
    // End-to-end tight-memory-pool parity test: build table has 4096 rows with 256-byte
    // payloads (~1 MiB), filter selects 16 rows (~4 KiB), and 128 KiB GreedyMemoryPool
    // verifies dead backing buffers are not retained in HashJoinBuildState and pool
    // reservations return to 0.
    let build_schema = Arc::new(Schema::new(vec![
        Field::new("bk", DataType::Int64, false),
        Field::new("gate", DataType::Int64, false),
        Field::new("wide", DataType::Utf8, false),
    ]));
    let n = 4096usize;
    let wide_str = "w".repeat(256);
    let build_batch = RecordBatch::try_new(
        Arc::clone(&build_schema),
        vec![
            Arc::new(Int64Array::from_iter_values(0..n as i64)),
            Arc::new(Int64Array::from_iter_values((0..n as i64).map(|i| i % 256))),
            Arc::new(StringArray::from(vec![wide_str.as_str(); n])),
        ],
    )?;
    let probe_schema = Arc::new(Schema::new(vec![
        Field::new("pk", DataType::Int64, false),
        Field::new("pval", DataType::Int64, false),
    ]));
    let probe_batch = RecordBatch::try_new(
        Arc::clone(&probe_schema),
        vec![
            Arc::new(Int64Array::from_iter_values(0..n as i64)),
            Arc::new(Int64Array::from_iter_values((0..n as i64).map(|i| i * 2))),
        ],
    )?;

    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(128 * 1024));
    let runtime = RuntimeEnvBuilder::new()
        .with_memory_pool(Arc::clone(&pool))
        .build_arc()?;
    let mut config = SessionConfig::new()
        .with_target_partitions(2)
        .with_batch_size(512);
    config.options_mut().execution.morsel_mode = MorselMode::Required;
    let state = datafusion::execution::SessionStateBuilder::new()
        .with_config(config)
        .with_runtime_env(runtime)
        .with_default_features()
        .build();
    let ctx = SessionContext::new_with_state(state);
    ctx.register_table(
        "b",
        Arc::new(MemTable::try_new(build_schema, vec![vec![build_batch]])?),
    )?;
    ctx.register_table(
        "p",
        Arc::new(MemTable::try_new(probe_schema, vec![vec![probe_batch]])?),
    )?;

    let sql = "SELECT p.pk, b.wide FROM p JOIN b ON p.pk = b.bk WHERE b.gate = 0";
    let plan = ctx.sql(sql).await?.create_physical_plan().await?;
    assert_eq!(plan.name(), "MorselExec");
    let out = collect(plan, ctx.task_ctx()).await?;
    let total: usize = out.iter().map(RecordBatch::num_rows).sum();
    assert_eq!(total, 16);
    assert_eq!(pool.reserved(), 0);
    Ok(())
}

#[cfg(feature = "parquet")]
#[tokio::test]
async fn parquet_filtered_join_group_by_and_projection_parity() -> Result<()> {
    use datafusion::prelude::ParquetReadOptions;
    use parquet::arrow::ArrowWriter;
    use parquet::file::properties::WriterProperties;
    use std::fs::File;
    use tempfile::TempDir;

    let dir = TempDir::new()?;
    let fact_dir = dir.path().join("fact");
    let dim_dir = dir.path().join("dim");
    std::fs::create_dir_all(&fact_dir)?;
    std::fs::create_dir_all(&dim_dir)?;

    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(64))
        .build();

    let fact_mem = sliced_fact_table()?;
    let mut fact_writer = ArrowWriter::try_new(
        File::create(fact_dir.join("part-0.parquet"))?,
        fact_mem.batches[0].read().await[0].schema(),
        Some(props.clone()),
    )?;
    for part in &fact_mem.batches {
        for b in part.read().await.iter() {
            if b.num_rows() > 0 {
                fact_writer.write(b)?;
            }
        }
    }
    fact_writer.close()?;

    let dim_mem = sliced_dim_table()?;
    let mut dim_writer = ArrowWriter::try_new(
        File::create(dim_dir.join("part-0.parquet"))?,
        dim_mem.batches[0].read().await[0].schema(),
        Some(props),
    )?;
    for part in &dim_mem.batches {
        for b in part.read().await.iter() {
            if b.num_rows() > 0 {
                dim_writer.write(b)?;
            }
        }
    }
    dim_writer.close()?;

    let queries = [
        "SELECT f.id, f.payload, d.weight FROM fact f JOIN dim d ON f.k = d.dk WHERE f.gate > 5",
        "SELECT d.dgrp, COUNT(*) AS n, SUM(f.val) AS s, AVG(f.fval) AS a FROM fact f JOIN dim d ON f.k = d.dk WHERE f.gate > 4 GROUP BY d.dgrp",
        "SELECT grp, COUNT(*) AS n, SUM(val) AS sv, AVG(fval) AS af FROM fact WHERE gate > 3 GROUP BY grp",
        "SELECT id, payload, val * 2 AS v2 FROM fact WHERE gate > 6",
        "SELECT id, payload FROM fact",
    ];

    for pushdown in [false, true] {
        let mut stock_cfg = SessionConfig::new()
            .with_target_partitions(4)
            .with_batch_size(32);
        stock_cfg.options_mut().execution.morsel_mode = MorselMode::Off;
        stock_cfg.options_mut().execution.parquet.pushdown_filters = pushdown;
        stock_cfg.options_mut().execution.collect_statistics = false;
        let stock_ctx = SessionContext::new_with_config(stock_cfg);

        let mut morsel_cfg = SessionConfig::new()
            .with_target_partitions(4)
            .with_batch_size(32);
        morsel_cfg.options_mut().execution.morsel_mode = MorselMode::Required;
        morsel_cfg.options_mut().execution.parquet.pushdown_filters = pushdown;
        morsel_cfg.options_mut().execution.collect_statistics = false;
        let morsel_ctx = SessionContext::new_with_config(morsel_cfg);

        for ctx in [&stock_ctx, &morsel_ctx] {
            ctx.register_parquet(
                "fact",
                fact_dir.to_str().unwrap(),
                ParquetReadOptions::default(),
            )
            .await?;
            ctx.register_parquet(
                "dim",
                dim_dir.to_str().unwrap(),
                ParquetReadOptions::default(),
            )
            .await?;
        }

        for sql in queries {
            let stock_plan = stock_ctx.sql(sql).await?.create_physical_plan().await?;
            let morsel_plan = morsel_ctx.sql(sql).await?.create_physical_plan().await?;
            assert_eq!(
                morsel_plan.name(),
                "MorselExec",
                "pushdown={pushdown} {sql}"
            );
            assert_eq!(morsel_plan.schema(), stock_plan.schema());
            let expected = collect(stock_plan, stock_ctx.task_ctx()).await?;
            let actual = collect(morsel_plan, morsel_ctx.task_ctx()).await?;
            assert_eq!(
                sorted_rows(&actual)?,
                sorted_rows(&expected)?,
                "pushdown={pushdown} {sql}"
            );
        }
    }
    Ok(())
}

#[tokio::test]
async fn direct_count_col_logical_nulls_for_null_type_and_dictionary_values() -> Result<()>
{
    // 1. Exact DataType::Null repro via CREATE TABLE t AS VALUES ...
    for workers in [1usize, 4usize] {
        let stock_ctx = session(MorselMode::Off, workers, 8);
        let morsel_ctx = session(MorselMode::Off, workers, 8);
        for ctx in [&stock_ctx, &morsel_ctx] {
            ctx.sql("CREATE TABLE t AS VALUES (1, NULL, 5), (1, NULL, 6), (2, NULL, 7)")
                .await?
                .collect()
                .await?;
        }
        morsel_ctx
            .sql("SET datafusion.execution.morsel_mode = 'required'")
            .await?
            .collect()
            .await?;
        for sql in [
            "SELECT column1, COUNT(column2) FROM t GROUP BY column1",
            "SELECT column1, COUNT(column2) FROM t WHERE column3 > 5 GROUP BY column1",
            "SELECT column1, COUNT(column2) FROM t WHERE column3 > 6 GROUP BY column1",
        ] {
            let stock_plan = stock_ctx.sql(sql).await?.create_physical_plan().await?;
            let morsel_plan = morsel_ctx.sql(sql).await?.create_physical_plan().await?;
            assert_eq!(morsel_plan.name(), "MorselExec", "workers={workers} {sql}");
            assert_eq!(morsel_plan.schema(), stock_plan.schema());
            let expected = collect(stock_plan, stock_ctx.task_ctx()).await?;
            let actual = collect(morsel_plan, morsel_ctx.task_ctx()).await?;
            assert_eq!(
                sorted_rows(&actual)?,
                sorted_rows(&expected)?,
                "workers={workers} {sql}"
            );
        }
    }

    // 2. MemTable with Dictionary(Int32, Utf8) whose dictionary values include NULL,
    //    grouped by an Int32 key.
    let dict_type =
        DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8));
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, true),
        Field::new("d", dict_type, true),
        Field::new("gate", DataType::Int32, true),
    ]));
    // Dictionary values: index 0 -> "a", index 1 -> NULL, index 2 -> "b".
    // Keys include valid keys pointing to NULL dict value (index 1) AND null keys.
    let dict_values = Arc::new(StringArray::from(vec![Some("a"), None, Some("b")]));
    let dict_keys = Int32Array::from(vec![
        Some(0), // k=1, "a"  (non-null)
        Some(1), // k=1, NULL (null in dict values!)
        None,    // k=1, NULL (null key)
        Some(2), // k=2, "b"  (non-null)
        Some(1), // k=2, NULL (null in dict values!)
        Some(1), // k=NULL, NULL (null in dict values!)
        Some(0), // k=NULL, "a" (non-null)
        Some(1), // k=3, NULL (all logically null for k=3!)
    ]);
    let dict_array = DictionaryArray::<Int32Type>::try_new(dict_keys, dict_values)?;
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![
                Some(1),
                Some(1),
                Some(1),
                Some(2),
                Some(2),
                None,
                None,
                Some(3),
            ])),
            Arc::new(dict_array),
            Arc::new(Int32Array::from(vec![
                Some(10),
                Some(20),
                Some(30),
                Some(5),
                Some(25),
                Some(15),
                None,
                Some(40),
            ])),
        ],
    )?;
    let table = Arc::new(MemTable::try_new(
        schema,
        vec![vec![batch.slice(0, 4)], vec![batch.slice(4, 4)]],
    )?);

    for workers in [1usize, 4usize] {
        let stock_ctx = session(MorselMode::Off, workers, 4);
        let morsel_ctx = session(MorselMode::Required, workers, 4);
        stock_ctx.register_table("dict_t", table.clone())?;
        morsel_ctx.register_table("dict_t", table.clone())?;

        for sql in [
            "SELECT k, COUNT(d), COUNT(*) FROM dict_t GROUP BY k",
            "SELECT k, COUNT(d), COUNT(*) FROM dict_t WHERE gate > 10 GROUP BY k",
            "SELECT k, COUNT(d), COUNT(*) FROM dict_t WHERE gate >= 10 AND gate <= 30 GROUP BY k",
        ] {
            let stock_plan = stock_ctx.sql(sql).await?.create_physical_plan().await?;
            let morsel_plan = morsel_ctx.sql(sql).await?.create_physical_plan().await?;
            assert_eq!(morsel_plan.name(), "MorselExec", "workers={workers} {sql}");
            assert_eq!(morsel_plan.schema(), stock_plan.schema());
            let expected = collect(stock_plan, stock_ctx.task_ctx()).await?;
            let actual = collect(morsel_plan, morsel_ctx.task_ctx()).await?;
            assert_eq!(
                sorted_rows(&actual)?,
                sorted_rows(&expected)?,
                "workers={workers} {sql}"
            );
        }
    }
    Ok(())
}

#[tokio::test]
async fn filtered_hash_join_probe_under_tight_memory_pool_parity() -> Result<()> {
    // End-to-end memory-limit parity test: build table is 16 rows and probe MemTable holds
    // one 4096-row batch with a 256-byte Utf8 column (~1.1 MiB preexisting storage).
    // Under a 512 KiB GreedyMemoryPool, unfiltered, 75%-filtered (`gate <> 0`),
    // 25%-filtered (`gate = 0`), and tiny-filtered (`pk % 256 = 0`) probes must all
    // succeed without charging preexisting MemTable storage and leave pool.reserved() == 0.
    // (Direct cursor compaction and exact reservation bytes are verified in
    // `morsel::hash_join::tests::probe_cursor_reservation_accounts_for_selection_and_compaction_only`.)
    let build_schema = Arc::new(Schema::new(vec![
        Field::new("bk", DataType::Int64, false),
        Field::new("bval", DataType::Int64, false),
    ]));
    let build_batch = RecordBatch::try_new(
        Arc::clone(&build_schema),
        vec![
            Arc::new(Int64Array::from_iter_values(0..16)),
            Arc::new(Int64Array::from_iter_values((0..16).map(|i| i * 10))),
        ],
    )?;

    let probe_schema = Arc::new(Schema::new(vec![
        Field::new("pk", DataType::Int64, false),
        Field::new("gate", DataType::Int64, false),
        Field::new("wide", DataType::Utf8, false),
    ]));
    let n = 4096usize;
    let wide_str = "p".repeat(256);
    let probe_batch = RecordBatch::try_new(
        Arc::clone(&probe_schema),
        vec![
            Arc::new(Int64Array::from_iter_values(0..n as i64)),
            Arc::new(Int64Array::from_iter_values((0..n as i64).map(|i| i % 4))),
            Arc::new(StringArray::from(vec![wide_str.as_str(); n])),
        ],
    )?;

    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(512 * 1024));
    let runtime = RuntimeEnvBuilder::new()
        .with_memory_pool(Arc::clone(&pool))
        .build_arc()?;
    let mut config = SessionConfig::new()
        .with_target_partitions(1)
        .with_batch_size(512);
    config.options_mut().execution.morsel_mode = MorselMode::Required;
    let state = datafusion::execution::SessionStateBuilder::new()
        .with_config(config)
        .with_runtime_env(runtime)
        .with_default_features()
        .build();
    let ctx = SessionContext::new_with_state(state);
    ctx.register_table(
        "b",
        Arc::new(MemTable::try_new(build_schema, vec![vec![build_batch]])?),
    )?;
    ctx.register_table(
        "p",
        Arc::new(MemTable::try_new(probe_schema, vec![vec![probe_batch]])?),
    )?;

    for (sql, expected_rows) in [
        (
            "SELECT p.pk, p.wide, b.bval FROM p JOIN b ON p.pk = b.bk",
            16,
        ),
        (
            "SELECT p.pk, p.wide, b.bval FROM p JOIN b ON p.pk = b.bk WHERE p.gate <> 0",
            12,
        ),
        (
            "SELECT p.pk, p.wide, b.bval FROM p JOIN b ON p.pk = b.bk WHERE p.gate = 0",
            4,
        ),
        (
            "SELECT p.pk, p.wide, b.bval FROM p JOIN b ON p.pk = b.bk WHERE p.pk % 256 = 0",
            1,
        ),
    ] {
        let plan = ctx.sql(sql).await?.create_physical_plan().await?;
        assert_eq!(plan.name(), "MorselExec", "{sql}");
        let out = collect(plan, ctx.task_ctx()).await?;
        let total: usize = out.iter().map(RecordBatch::num_rows).sum();
        assert_eq!(total, expected_rows, "{sql}");
        assert_eq!(pool.reserved(), 0, "{sql}");
    }
    Ok(())
}
