// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! End-to-end TPC-H 22/22 native morsel coverage and multi-stage DAG parity tests.

#![cfg(all(feature = "sql", feature = "parquet"))]

use std::fs::File;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use arrow::array::{
    ArrayRef, Date32Array, Decimal128Array, Int32Array, Int64Array, RecordBatch,
    StringArray,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::row::{OwnedRow, RowConverter, SortField};
use datafusion::common::config::MorselMode;
use datafusion::datasource::MemTable;
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::SessionStateBuilder;
use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryPool};
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::logical_expr::{Distinct, LogicalPlan};
use datafusion::morsel::{MorselMetrics, MorselQueryPlanner};
use datafusion::physical_plan::{collect, displayable, execute_stream};
use datafusion::prelude::{ParquetReadOptions, SessionConfig, SessionContext};
use futures::StreamExt;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use tempfile::TempDir;
use tokio::time::timeout;

fn days(year: i32, month: u32, day: u32) -> i32 {
    // Civil date to days since 1970-01-01 (Howard Hinnant's algorithm).
    let y = year - i32::from(month <= 2);
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400) as u32;
    let m = month;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe as i32 - 719_468
}

fn dec(val: i128) -> i128 {
    val
}

fn write_table_parquet(dir: &TempDir, name: &str, batch: &RecordBatch) -> Result<String> {
    let table_dir = dir.path().join(name);
    std::fs::create_dir_all(&table_dir)?;
    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(4))
        .build();
    let mid = batch.num_rows() / 2;
    let parts: Vec<RecordBatch> = if batch.num_rows() >= 2 {
        vec![
            batch.slice(0, mid),
            batch.slice(mid, batch.num_rows() - mid),
        ]
    } else {
        vec![batch.clone()]
    };
    for (idx, part) in parts.iter().enumerate() {
        let path = table_dir.join(format!("{idx}.parquet"));
        let mut writer = ArrowWriter::try_new(
            File::create(path)?,
            part.schema(),
            Some(props.clone()),
        )?;
        writer.write(part)?;
        writer.close()?;
    }
    Ok(table_dir.to_string_lossy().into_owned())
}

fn build_tpch_parquet_fixture() -> Result<(TempDir, Vec<(&'static str, String)>)> {
    let dir = TempDir::new()?;

    // region
    let region_schema = Arc::new(Schema::new(vec![
        Field::new("r_regionkey", DataType::Int64, false),
        Field::new("r_name", DataType::Utf8, false),
        Field::new("r_comment", DataType::Utf8, false),
    ]));
    let region = RecordBatch::try_new(
        region_schema,
        vec![
            Arc::new(Int64Array::from(vec![0, 1, 2, 3, 4])) as ArrayRef,
            Arc::new(StringArray::from(vec![
                "AFRICA",
                "AMERICA",
                "ASIA",
                "EUROPE",
                "MIDDLE EAST",
            ])),
            Arc::new(StringArray::from(vec!["r0", "r1", "r2", "r3", "r4"])),
        ],
    )?;

    // nation
    let nation_schema = Arc::new(Schema::new(vec![
        Field::new("n_nationkey", DataType::Int64, false),
        Field::new("n_name", DataType::Utf8, false),
        Field::new("n_regionkey", DataType::Int64, false),
        Field::new("n_comment", DataType::Utf8, false),
    ]));
    let nation = RecordBatch::try_new(
        nation_schema,
        vec![
            Arc::new(Int64Array::from(vec![0, 1, 2, 3, 6, 7, 8, 9, 20])) as ArrayRef,
            Arc::new(StringArray::from(vec![
                "ALGERIA",
                "ARGENTINA",
                "BRAZIL",
                "CANADA",
                "FRANCE",
                "GERMANY",
                "INDIA",
                "INDONESIA",
                "SAUDI ARABIA",
            ])),
            Arc::new(Int64Array::from(vec![0, 1, 1, 1, 3, 3, 2, 2, 4])),
            Arc::new(StringArray::from(vec![
                "n0", "n1", "n2", "n3", "n6", "n7", "n8", "n9", "n20",
            ])),
        ],
    )?;

    // part
    let part_schema = Arc::new(Schema::new(vec![
        Field::new("p_partkey", DataType::Int64, false),
        Field::new("p_name", DataType::Utf8, false),
        Field::new("p_mfgr", DataType::Utf8, false),
        Field::new("p_brand", DataType::Utf8, false),
        Field::new("p_type", DataType::Utf8, false),
        Field::new("p_size", DataType::Int32, false),
        Field::new("p_container", DataType::Utf8, false),
        Field::new("p_retailprice", DataType::Decimal128(15, 2), false),
        Field::new("p_comment", DataType::Utf8, false),
    ]));
    let part = RecordBatch::try_new(
        part_schema,
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5, 6])) as ArrayRef,
            Arc::new(StringArray::from(vec![
                "forest green brass",
                "forest steel green",
                "promo copper",
                "shiny green steel",
                "excluded part",
                "second brass",
            ])),
            Arc::new(StringArray::from(vec![
                "Manufacturer#1",
                "Manufacturer#2",
                "Manufacturer#3",
                "Manufacturer#4",
                "Manufacturer#5",
                "Manufacturer#1",
            ])),
            Arc::new(StringArray::from(vec![
                "Brand#12", "Brand#23", "Brand#12", "Brand#34", "Brand#45", "Brand#23",
            ])),
            Arc::new(StringArray::from(vec![
                "LARGE BRASS",
                "ECONOMY ANODIZED STEEL",
                "PROMO BURNISHED COPPER",
                "STANDARD POLISHED BRASS",
                "MEDIUM POLISHED TIN",
                "SMALL BRASS",
            ])),
            Arc::new(Int32Array::from(vec![15, 9, 3, 14, 15, 15])),
            Arc::new(StringArray::from(vec![
                "SM CASE", "MED BOX", "SM BOX", "LG BOX", "LG PKG", "MED BOX",
            ])),
            Arc::new(
                Decimal128Array::from(vec![
                    dec(10000),
                    dec(20000),
                    dec(15000),
                    dec(30000),
                    dec(5000),
                    dec(12000),
                ])
                .with_precision_and_scale(15, 2)?,
            ),
            Arc::new(StringArray::from(vec!["p1", "p2", "p3", "p4", "p5", "p6"])),
        ],
    )?;

    // supplier
    let supplier_schema = Arc::new(Schema::new(vec![
        Field::new("s_suppkey", DataType::Int64, false),
        Field::new("s_name", DataType::Utf8, false),
        Field::new("s_address", DataType::Utf8, false),
        Field::new("s_nationkey", DataType::Int64, false),
        Field::new("s_phone", DataType::Utf8, false),
        Field::new("s_acctbal", DataType::Decimal128(15, 2), false),
        Field::new("s_comment", DataType::Utf8, false),
    ]));
    let supplier = RecordBatch::try_new(
        supplier_schema,
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8])) as ArrayRef,
            Arc::new(StringArray::from(vec![
                "Supplier#000000001",
                "Supplier#000000002",
                "Supplier#000000003",
                "Supplier#000000004",
                "Supplier#000000005",
                "Supplier#000000006",
                "Supplier#000000007",
                "Supplier#000000008",
            ])),
            Arc::new(StringArray::from(vec![
                "Addr1", "Addr2", "Addr3", "Addr4", "Addr5", "Addr6", "Addr7", "Addr8",
            ])),
            Arc::new(Int64Array::from(vec![7, 6, 8, 2, 3, 20, 7, 9])),
            Arc::new(StringArray::from(vec![
                "17-100-100-1001",
                "16-200-200-2002",
                "18-300-300-3003",
                "12-400-400-4004",
                "13-500-500-5005",
                "30-600-600-6006",
                "17-700-700-7007",
                "19-800-800-8008",
            ])),
            Arc::new(
                Decimal128Array::from(vec![
                    dec(500000),
                    dec(700000),
                    dec(300000),
                    dec(450000),
                    dec(600000),
                    dec(250000),
                    dec(100000),
                    dec(800000),
                ])
                .with_precision_and_scale(15, 2)?,
            ),
            Arc::new(StringArray::from(vec![
                "regular supplier",
                "regular supplier",
                "regular supplier",
                "regular supplier",
                "regular supplier",
                "regular supplier",
                "has Customer Complaints here",
                "regular supplier",
            ])),
        ],
    )?;

    // partsupp
    let partsupp_schema = Arc::new(Schema::new(vec![
        Field::new("ps_partkey", DataType::Int64, false),
        Field::new("ps_suppkey", DataType::Int64, false),
        Field::new("ps_availqty", DataType::Int32, false),
        Field::new("ps_supplycost", DataType::Decimal128(15, 2), false),
        Field::new("ps_comment", DataType::Utf8, false),
    ]));
    let partsupp = RecordBatch::try_new(
        partsupp_schema,
        vec![
            Arc::new(Int64Array::from(vec![
                1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 6, 6,
            ])) as ArrayRef,
            Arc::new(Int64Array::from(vec![
                1, 2, 5, 1, 2, 4, 5, 7, 1, 2, 3, 1, 2, 7, 1, 2,
            ])),
            Arc::new(Int32Array::from(vec![
                100, 200, 500, 300, 150, 120, 400, 50, 50, 80, 90, 600, 100, 20, 100, 100,
            ])),
            Arc::new(
                Decimal128Array::from(vec![
                    dec(5000),
                    dec(4000),
                    dec(6000),
                    dec(8000),
                    dec(7500),
                    dec(6000),
                    dec(7000),
                    dec(3000),
                    dec(1000),
                    dec(1200),
                    dec(1500),
                    dec(20000),
                    dec(9000),
                    dec(2000),
                    dec(2500),
                    dec(2500),
                ])
                .with_precision_and_scale(15, 2)?,
            ),
            Arc::new(StringArray::from(vec!["ps"; 16])),
        ],
    )?;

    // customer
    let customer_schema = Arc::new(Schema::new(vec![
        Field::new("c_custkey", DataType::Int64, false),
        Field::new("c_name", DataType::Utf8, false),
        Field::new("c_address", DataType::Utf8, false),
        Field::new("c_nationkey", DataType::Int64, false),
        Field::new("c_phone", DataType::Utf8, false),
        Field::new("c_acctbal", DataType::Decimal128(15, 2), false),
        Field::new("c_mktsegment", DataType::Utf8, false),
        Field::new("c_comment", DataType::Utf8, false),
    ]));
    let customer = RecordBatch::try_new(
        customer_schema,
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8])) as ArrayRef,
            Arc::new(StringArray::from(vec![
                "Customer#000000001",
                "Customer#000000002",
                "Customer#000000003",
                "Customer#000000004",
                "Customer#000000005",
                "Customer#000000006",
                "Customer#000000007",
                "Customer#000000008",
            ])),
            Arc::new(StringArray::from(vec![
                "CAddr1", "CAddr2", "CAddr3", "CAddr4", "CAddr5", "CAddr6", "CAddr7",
                "CAddr8",
            ])),
            Arc::new(Int64Array::from(vec![6, 7, 8, 2, 3, 1, 20, 9])),
            Arc::new(StringArray::from(vec![
                "13-111-222-3333",
                "31-222-333-4444",
                "23-333-444-5555",
                "29-444-555-6666",
                "30-555-666-7777",
                "18-666-777-8888",
                "17-777-888-9999",
                "13-888-999-0000",
            ])),
            Arc::new(
                Decimal128Array::from(vec![
                    dec(100000),
                    dec(200000),
                    dec(300000),
                    dec(50000),
                    dec(500000),
                    dec(600000),
                    dec(10000),
                    dec(-5000),
                ])
                .with_precision_and_scale(15, 2)?,
            ),
            Arc::new(StringArray::from(vec![
                "BUILDING",
                "BUILDING",
                "AUTOMOBILE",
                "BUILDING",
                "MACHINERY",
                "FURNITURE",
                "HOUSEHOLD",
                "BUILDING",
            ])),
            Arc::new(StringArray::from(vec!["c"; 8])),
        ],
    )?;

    // orders
    let orders_schema = Arc::new(Schema::new(vec![
        Field::new("o_orderkey", DataType::Int64, false),
        Field::new("o_custkey", DataType::Int64, false),
        Field::new("o_orderstatus", DataType::Utf8, false),
        Field::new("o_totalprice", DataType::Decimal128(15, 2), false),
        Field::new("o_orderdate", DataType::Date32, false),
        Field::new("o_orderpriority", DataType::Utf8, false),
        Field::new("o_clerk", DataType::Utf8, false),
        Field::new("o_shippriority", DataType::Int32, false),
        Field::new("o_comment", DataType::Utf8, false),
    ]));
    let orders = RecordBatch::try_new(
        orders_schema,
        vec![
            Arc::new(Int64Array::from(vec![
                101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112,
            ])) as ArrayRef,
            Arc::new(Int64Array::from(vec![1, 2, 3, 4, 1, 2, 1, 4, 3, 1, 2, 8])),
            Arc::new(StringArray::from(vec![
                "O", "O", "F", "O", "F", "F", "O", "O", "F", "F", "O", "F",
            ])),
            Arc::new(
                Decimal128Array::from(vec![
                    dec(500000),
                    dec(600000),
                    dec(700000),
                    dec(800000),
                    dec(900000),
                    dec(400000),
                    dec(300000),
                    dec(200000),
                    dec(150000),
                    dec(950000),
                    dec(250000),
                    dec(180000),
                ])
                .with_precision_and_scale(15, 2)?,
            ),
            Arc::new(Date32Array::from(vec![
                days(1993, 7, 15),  // 101: Q4
                days(1993, 10, 15), // 102: Q10
                days(1994, 5, 1),   // 103: Q5 (ASIA cust 3)
                days(1995, 6, 1),   // 104: Q8 (1995, AMERICA cust 4)
                days(1995, 2, 1),   // 105: Q3 (< 1995-03-15, BUILDING cust 1), Q21 (F)
                days(1996, 2, 1),   // 106: Q15, Q7, Q8 (1996), Q21 (F, rejected by l3)
                days(1995, 9, 10),  // 107: Q14
                days(1996, 6, 15),  // 108: Q8 (1996, AMERICA cust 4)
                days(1994, 3, 10),  // 109: Q12, Q6
                days(1994, 8, 20),  // 110: Q18 (sum qty > 300)
                days(1993, 8, 10),  // 111: Q4, Q13 (special requests)
                days(1994, 11, 5),  // 112: Q3 (< 1995-03-15)
            ])),
            Arc::new(StringArray::from(vec![
                "1-URGENT",
                "2-HIGH",
                "3-MEDIUM",
                "1-URGENT",
                "2-HIGH",
                "4-NOT SPECIFIED",
                "5-LOW",
                "1-URGENT",
                "2-HIGH",
                "1-URGENT",
                "3-MEDIUM",
                "2-HIGH",
            ])),
            Arc::new(StringArray::from(vec!["Clerk#1"; 12])),
            Arc::new(Int32Array::from(vec![0; 12])),
            Arc::new(StringArray::from(vec![
                "normal",
                "normal",
                "normal",
                "normal",
                "normal",
                "normal",
                "normal",
                "normal",
                "normal",
                "normal",
                "has special requests inside",
                "normal",
            ])),
        ],
    )?;

    // lineitem
    // Columns: orderkey, partkey, suppkey, linenumber, qty, price, disc, tax,
    //          returnflag, linestatus, shipdate, commitdate, receiptdate,
    //          shipinstruct, shipmode, comment
    type LineitemRow<'a> = (
        i64,
        i64,
        i64,
        i32,
        i128,
        i128,
        i128,
        i128,
        &'a str,
        &'a str,
        i32,
        i32,
        i32,
        &'a str,
        &'a str,
    );
    let lineitems: Vec<LineitemRow<'_>> = vec![
        // 101 (Q4: commit < receipt, Q1)
        (
            101,
            1,
            1,
            1,
            dec(1000),
            dec(100000),
            dec(5),
            dec(2),
            "A",
            "F",
            days(1993, 8, 1),
            days(1993, 8, 10),
            days(1993, 8, 20),
            "DELIVER IN PERSON",
            "AIR",
        ),
        // 102 (Q10: returnflag='R')
        (
            102,
            2,
            2,
            1,
            dec(1500),
            dec(200000),
            dec(6),
            dec(3),
            "R",
            "F",
            days(1993, 11, 1),
            days(1993, 11, 5),
            days(1993, 11, 10),
            "DELIVER IN PERSON",
            "AIR REG",
        ),
        // 103 (Q5: cust 3 INDIA, supp 3 INDIA, 1994; Q6: shipdate 1994, disc 0.06, qty < 24)
        (
            103,
            3,
            3,
            1,
            dec(1000),
            dec(150000),
            dec(6),
            dec(4),
            "N",
            "O",
            days(1994, 6, 1),
            days(1994, 6, 10),
            days(1994, 6, 20),
            "DELIVER IN PERSON",
            "MAIL",
        ),
        // 104 (Q8: 1995, AMERICA cust 4, BRAZIL supp 4, part 2 ECONOMY ANODIZED STEEL; Q17: qty=1.00)
        (
            104,
            2,
            4,
            1,
            dec(100),
            dec(50000),
            dec(5),
            dec(2),
            "N",
            "O",
            days(1995, 7, 1),
            days(1995, 7, 10),
            days(1995, 7, 15),
            "DELIVER IN PERSON",
            "SHIP",
        ),
        // 104 line 2 (Q8: 1995, AMERICA cust 4, non-BRAZIL supp 1, part 2; Q17: qty=50.00 -> avg=25.5, 0.2*avg=5.1)
        (
            104,
            2,
            1,
            2,
            dec(5000),
            dec(400000),
            dec(4),
            dec(2),
            "N",
            "O",
            days(1995, 7, 2),
            days(1995, 7, 10),
            days(1995, 7, 15),
            "NONE",
            "TRUCK",
        ),
        // 105 (Q3: o_orderdate < 1995-03-15, l_shipdate > 1995-03-15; Q7: cust 1 FRANCE, supp 1 GERMANY, ship 1995; Q21: supp 6 late + supp 8 on-time)
        (
            105,
            1,
            1,
            1,
            dec(800),
            dec(120000),
            dec(5),
            dec(2),
            "N",
            "O",
            days(1995, 4, 1),
            days(1995, 4, 12),
            days(1995, 4, 10),
            "DELIVER IN PERSON",
            "AIR",
        ),
        // 105 line 2 (Q21: SAUDI ARABIA supp 6, receipt > commit)
        (
            105,
            4,
            6,
            2,
            dec(1200),
            dec(220000),
            dec(3),
            dec(1),
            "N",
            "O",
            days(1995, 4, 2),
            days(1995, 4, 5),
            days(1995, 4, 15),
            "NONE",
            "RAIL",
        ),
        // 105 line 3 (Q21: other supp 8, receipt <= commit -> passes EXISTS and NOT EXISTS!)
        (
            105,
            4,
            8,
            3,
            dec(900),
            dec(110000),
            dec(2),
            dec(1),
            "N",
            "O",
            days(1995, 4, 2),
            days(1995, 4, 12),
            days(1995, 4, 10),
            "NONE",
            "RAIL",
        ),
        // 106 (Q7: cust 2 GERMANY, supp 2 FRANCE, ship 1996; Q15: shipdate in 1996-01..03; Q21 rejected order: both supp 6 & supp 2 late)
        (
            106,
            4,
            2,
            1,
            dec(2500),
            dec(600000),
            dec(5),
            dec(2),
            "A",
            "F",
            days(1996, 2, 15),
            days(1996, 2, 20),
            days(1996, 2, 25),
            "DELIVER IN PERSON",
            "AIR",
        ),
        (
            106,
            4,
            6,
            2,
            dec(2200),
            dec(180000),
            dec(5),
            dec(2),
            "A",
            "F",
            days(1996, 2, 16),
            days(1996, 2, 20),
            days(1996, 2, 28),
            "DELIVER IN PERSON",
            "AIR REG",
        ),
        // 107 (Q14: shipdate in 1995-09, PROMO part 3 & non-PROMO part 2; Q19 branch 1 & 2)
        (
            107,
            3,
            1,
            1,
            dec(500),
            dec(90000),
            dec(5),
            dec(2),
            "N",
            "O",
            days(1995, 9, 15),
            days(1995, 9, 20),
            days(1995, 9, 25),
            "DELIVER IN PERSON",
            "AIR",
        ),
        (
            107,
            2,
            2,
            2,
            dec(1200),
            dec(180000),
            dec(5),
            dec(2),
            "N",
            "O",
            days(1995, 9, 16),
            days(1995, 9, 20),
            days(1995, 9, 25),
            "DELIVER IN PERSON",
            "AIR REG",
        ),
        // 108 (Q8: 1996, AMERICA cust 4, BRAZIL supp 4, part 2)
        (
            108,
            2,
            4,
            1,
            dec(1500),
            dec(250000),
            dec(5),
            dec(2),
            "N",
            "O",
            days(1996, 7, 1),
            days(1996, 7, 10),
            days(1996, 7, 15),
            "DELIVER IN PERSON",
            "AIR",
        ),
        // 109 (Q12: MAIL & SHIP, ship < commit < receipt, receipt in 1994; Q20: part 1, supp 5 CANADA, ship 1994)
        (
            109,
            1,
            5,
            1,
            dec(1000),
            dec(130000),
            dec(6),
            dec(2),
            "R",
            "F",
            days(1994, 4, 1),
            days(1994, 4, 10),
            days(1994, 4, 20),
            "DELIVER IN PERSON",
            "SHIP",
        ),
        // 110 (Q18: SUM(l_quantity) = 350 > 300; Q19 branch 3: Brand#34, LG BOX, qty 20..30)
        (
            110,
            4,
            1,
            1,
            dec(20000),
            dec(500000),
            dec(4),
            dec(2),
            "N",
            "O",
            days(1994, 9, 1),
            days(1994, 9, 10),
            days(1994, 9, 20),
            "NONE",
            "TRUCK",
        ),
        (
            110,
            4,
            2,
            2,
            dec(15000),
            dec(350000),
            dec(4),
            dec(2),
            "N",
            "O",
            days(1994, 9, 2),
            days(1994, 9, 10),
            days(1994, 9, 20),
            "NONE",
            "TRUCK",
        ),
        // 111 (Q4, Q19 branch 3: qty=25)
        (
            111,
            4,
            1,
            1,
            dec(2500),
            dec(240000),
            dec(5),
            dec(2),
            "A",
            "F",
            days(1993, 9, 1),
            days(1993, 9, 5),
            days(1993, 9, 15),
            "DELIVER IN PERSON",
            "AIR",
        ),
        // 112 (Q15: second supplier in 1996-01..03 with smaller revenue than supp 2)
        (
            112,
            2,
            1,
            1,
            dec(500),
            dec(50000),
            dec(5),
            dec(2),
            "N",
            "O",
            days(1996, 1, 20),
            days(1996, 1, 25),
            days(1996, 1, 30),
            "NONE",
            "MAIL",
        ),
    ];

    let lineitem_schema = Arc::new(Schema::new(vec![
        Field::new("l_orderkey", DataType::Int64, false),
        Field::new("l_partkey", DataType::Int64, false),
        Field::new("l_suppkey", DataType::Int64, false),
        Field::new("l_linenumber", DataType::Int32, false),
        Field::new("l_quantity", DataType::Decimal128(15, 2), false),
        Field::new("l_extendedprice", DataType::Decimal128(15, 2), false),
        Field::new("l_discount", DataType::Decimal128(15, 2), false),
        Field::new("l_tax", DataType::Decimal128(15, 2), false),
        Field::new("l_returnflag", DataType::Utf8, false),
        Field::new("l_linestatus", DataType::Utf8, false),
        Field::new("l_shipdate", DataType::Date32, false),
        Field::new("l_commitdate", DataType::Date32, false),
        Field::new("l_receiptdate", DataType::Date32, false),
        Field::new("l_shipinstruct", DataType::Utf8, false),
        Field::new("l_shipmode", DataType::Utf8, false),
        Field::new("l_comment", DataType::Utf8, false),
    ]));
    let lineitem = RecordBatch::try_new(
        lineitem_schema,
        vec![
            Arc::new(Int64Array::from_iter_values(lineitems.iter().map(|r| r.0)))
                as ArrayRef,
            Arc::new(Int64Array::from_iter_values(lineitems.iter().map(|r| r.1))),
            Arc::new(Int64Array::from_iter_values(lineitems.iter().map(|r| r.2))),
            Arc::new(Int32Array::from_iter_values(lineitems.iter().map(|r| r.3))),
            Arc::new(
                Decimal128Array::from_iter_values(lineitems.iter().map(|r| r.4))
                    .with_precision_and_scale(15, 2)?,
            ),
            Arc::new(
                Decimal128Array::from_iter_values(lineitems.iter().map(|r| r.5))
                    .with_precision_and_scale(15, 2)?,
            ),
            Arc::new(
                Decimal128Array::from_iter_values(lineitems.iter().map(|r| r.6))
                    .with_precision_and_scale(15, 2)?,
            ),
            Arc::new(
                Decimal128Array::from_iter_values(lineitems.iter().map(|r| r.7))
                    .with_precision_and_scale(15, 2)?,
            ),
            Arc::new(StringArray::from_iter_values(lineitems.iter().map(|r| r.8))),
            Arc::new(StringArray::from_iter_values(lineitems.iter().map(|r| r.9))),
            Arc::new(Date32Array::from_iter_values(
                lineitems.iter().map(|r| r.10),
            )),
            Arc::new(Date32Array::from_iter_values(
                lineitems.iter().map(|r| r.11),
            )),
            Arc::new(Date32Array::from_iter_values(
                lineitems.iter().map(|r| r.12),
            )),
            Arc::new(StringArray::from_iter_values(
                lineitems.iter().map(|r| r.13),
            )),
            Arc::new(StringArray::from_iter_values(
                lineitems.iter().map(|r| r.14),
            )),
            Arc::new(StringArray::from(vec!["li"; lineitems.len()])),
        ],
    )?;

    let paths = vec![
        ("region", write_table_parquet(&dir, "region", &region)?),
        ("nation", write_table_parquet(&dir, "nation", &nation)?),
        ("part", write_table_parquet(&dir, "part", &part)?),
        (
            "supplier",
            write_table_parquet(&dir, "supplier", &supplier)?,
        ),
        (
            "partsupp",
            write_table_parquet(&dir, "partsupp", &partsupp)?,
        ),
        (
            "customer",
            write_table_parquet(&dir, "customer", &customer)?,
        ),
        ("orders", write_table_parquet(&dir, "orders", &orders)?),
        (
            "lineitem",
            write_table_parquet(&dir, "lineitem", &lineitem)?,
        ),
    ];
    Ok((dir, paths))
}

async fn register_tpch_tables(
    ctx: &SessionContext,
    paths: &[(&'static str, String)],
) -> Result<()> {
    for (name, path) in paths {
        ctx.register_parquet(*name, path, ParquetReadOptions::default())
            .await?;
    }
    Ok(())
}

fn stock_context(workers: usize, batch_size: usize) -> SessionContext {
    let mut config = SessionConfig::new()
        .with_target_partitions(workers)
        .with_batch_size(batch_size);
    config.options_mut().execution.morsel_mode = MorselMode::Off;
    SessionContext::new_with_config(config)
}

fn morsel_context(
    workers: usize,
    batch_size: usize,
    mode: MorselMode,
    pushdown: bool,
    pool: Option<Arc<dyn MemoryPool>>,
) -> SessionContext {
    let mut config = SessionConfig::new()
        .with_target_partitions(workers)
        .with_batch_size(batch_size);
    config.options_mut().execution.morsel_mode = mode;
    config.options_mut().execution.parquet.pushdown_filters = pushdown;
    config.options_mut().execution.parquet.reorder_filters = pushdown;
    let mut builder = SessionStateBuilder::new_with_default_features()
        .with_config(config)
        .with_query_planner(Arc::new(MorselQueryPlanner::new(
            NonZeroUsize::new(workers).unwrap(),
        )));
    if let Some(pool) = pool {
        let runtime = RuntimeEnvBuilder::new()
            .with_memory_pool(pool)
            .build_arc()
            .unwrap();
        builder = builder.with_runtime_env(runtime);
    }
    SessionContext::new_with_state(builder.build())
}

fn load_tpch_sql(q: usize) -> Vec<String> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let path = manifest_dir
        .join("../../benchmarks/queries")
        .join(format!("q{q}.sql"));
    let content = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
    content
        .split(';')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .collect()
}

fn to_rows(batches: &[RecordBatch], sort: bool) -> Result<Vec<OwnedRow>> {
    let Some(first) = batches.iter().find(|b| b.num_rows() > 0) else {
        return Ok(Vec::new());
    };
    let fields = first
        .schema()
        .fields()
        .iter()
        .map(|f| SortField::new(f.data_type().clone()))
        .collect();
    let converter = RowConverter::new(fields)?;
    let mut rows = Vec::new();
    for batch in batches {
        if batch.num_rows() == 0 {
            continue;
        }
        let converted = converter.convert_columns(batch.columns())?;
        rows.extend(converted.iter().map(|r| r.owned()));
    }
    if sort {
        rows.sort();
    }
    Ok(rows)
}

fn is_ordered_tpch(q: usize) -> bool {
    !matches!(q, 6 | 14 | 17 | 19)
}

#[tokio::test]
async fn tpch_all_22_queries_native_required_parity() -> Result<()> {
    let (_dir, paths) = build_tpch_parquet_fixture()?;

    for (workers, pushdown) in [(4, false), (4, true), (1, true)] {
        let stock = stock_context(workers, 4);
        let native = morsel_context(workers, 4, MorselMode::Required, pushdown, None);
        register_tpch_tables(&stock, &paths).await?;
        register_tpch_tables(&native, &paths).await?;

        for q in 1..=22 {
            let statements = load_tpch_sql(q);
            let mut select_ran = false;
            for stmt in &statements {
                let lower = stmt.to_ascii_lowercase();
                if lower.starts_with("create view") || lower.starts_with("drop view") {
                    stock.sql(stmt).await?.collect().await?;
                    native.sql(stmt).await?.collect().await?;
                    continue;
                }
                select_ran = true;

                // Assert EXPLAIN shows MorselExec (zero fallback!).
                let explain_batches = native
                    .sql(&format!("EXPLAIN {stmt}"))
                    .await?
                    .collect()
                    .await?;
                let explain_str = format!("{explain_batches:?}");
                assert!(
                    explain_str.contains("MorselExec"),
                    "TPC-H Q{q} (workers={workers}, pushdown={pushdown}) EXPLAIN missing MorselExec:\n{explain_str}"
                );

                let stock_plan = stock.sql(stmt).await?.create_physical_plan().await?;
                let native_plan = native.sql(stmt).await?.create_physical_plan().await?;
                assert_eq!(
                    native_plan.name(),
                    "MorselExec",
                    "TPC-H Q{q} expected MorselExec, got:\n{}",
                    displayable(native_plan.as_ref()).indent(true)
                );

                let expected = collect(Arc::clone(&stock_plan), stock.task_ctx()).await?;
                let actual = collect(Arc::clone(&native_plan), native.task_ctx()).await?;

                assert_eq!(
                    native_plan.schema().fields().len(),
                    stock_plan.schema().fields().len(),
                    "TPC-H Q{q} schema column count mismatch"
                );
                for (nf, sf) in native_plan
                    .schema()
                    .fields()
                    .iter()
                    .zip(stock_plan.schema().fields().iter())
                {
                    assert_eq!(nf.name(), sf.name(), "TPC-H Q{q} field name mismatch");
                    assert_eq!(
                        nf.data_type(),
                        sf.data_type(),
                        "TPC-H Q{q} field type mismatch"
                    );
                }

                let ordered = is_ordered_tpch(q);
                let expected_rows = to_rows(&expected, !ordered)?;
                let actual_rows = to_rows(&actual, !ordered)?;
                assert!(
                    !expected_rows.is_empty(),
                    "TPC-H Q{q} synthetic fixture should produce non-empty results"
                );
                assert_eq!(
                    actual_rows, expected_rows,
                    "TPC-H Q{q} (workers={workers}, pushdown={pushdown}) row/order parity mismatch"
                );
            }
            assert!(select_ran, "TPC-H Q{q} had no SELECT statement");
        }
    }
    Ok(())
}

#[tokio::test]
async fn multi_stage_dag_lifecycle_cancellation_and_oom_cleanup() -> Result<()> {
    let (_dir, paths) = build_tpch_parquet_fixture()?;

    // 1. 4-stage DAG: Aggregate -> Filter(HAVING) -> Join -> Aggregate -> TopN
    let sql = "
        SELECT
            sub.n_name,
            COUNT(*) AS order_cnt,
            SUM(sub.total_qty) AS grand_qty
        FROM (
            SELECT
                o.o_orderkey,
                n.n_name,
                SUM(l.l_quantity) AS total_qty
            FROM orders o
            JOIN customer c ON o.o_custkey = c.c_custkey
            JOIN nation n ON c.c_nationkey = n.n_nationkey
            JOIN (
                SELECT l_orderkey, SUM(l_quantity) AS l_quantity
                FROM lineitem
                GROUP BY l_orderkey
                HAVING SUM(l_quantity) > 5
            ) l ON o.o_orderkey = l.l_orderkey
            GROUP BY o.o_orderkey, n.n_name
            HAVING SUM(l.l_quantity) > 5
        ) sub
        GROUP BY sub.n_name
        ORDER BY grand_qty DESC, sub.n_name ASC
        LIMIT 3
    ";

    let stock = stock_context(4, 4);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(64 * 1024 * 1024));
    let metrics_collector = Arc::new(MorselMetrics::default());
    let mut cfg = SessionConfig::new()
        .with_target_partitions(4)
        .with_batch_size(4)
        .with_extension(Arc::clone(&metrics_collector));
    cfg.options_mut().execution.morsel_mode = MorselMode::Required;
    cfg.options_mut().execution.parquet.pushdown_filters = true;
    cfg.options_mut().execution.parquet.reorder_filters = true;
    let runtime = RuntimeEnvBuilder::new()
        .with_memory_pool(Arc::clone(&pool))
        .build_arc()?;
    let native = SessionContext::new_with_state(
        SessionStateBuilder::new_with_default_features()
            .with_config(cfg)
            .with_runtime_env(runtime)
            .with_query_planner(Arc::new(MorselQueryPlanner::new(
                NonZeroUsize::new(4).unwrap(),
            )))
            .build(),
    );
    register_tpch_tables(&stock, &paths).await?;
    register_tpch_tables(&native, &paths).await?;

    let expected = stock.sql(sql).await?.collect().await?;
    let plan = native.sql(sql).await?.create_physical_plan().await?;
    assert_eq!(plan.name(), "MorselExec");
    let display = displayable(plan.as_ref()).indent(false).to_string();
    assert!(
        display.contains("[Stage ") && display.contains("TopNSink"),
        "expected multi-stage DAG display, got:\n{display}"
    );
    let actual = collect(Arc::clone(&plan), native.task_ctx()).await?;
    assert_eq!(to_rows(&actual, false)?, to_rows(&expected, false)?);
    assert!(!to_rows(&actual, false)?.is_empty());
    assert_eq!(
        pool.reserved(),
        0,
        "memory leaked after multi-stage DAG run"
    );
    let runs = metrics_collector.take_runs();
    assert_eq!(runs.len(), 1);
    assert!(
        runs[0]
            .sum_by_name("morsels_completed")
            .is_some_and(|v| v.as_usize() > 0),
        "expected per-pipeline morsel metrics"
    );

    // 2. Mid-DAG early drop / cancellation releases all reservations.
    let mut stream = execute_stream(Arc::clone(&plan), native.task_ctx())?;
    let _ = futures::poll!(stream.next());
    drop(stream);
    timeout(Duration::from_secs(5), async {
        while pool.reserved() != 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("pool reservation leaked after mid-DAG cancellation");

    // 3. Tight memory pool triggers clean ResourcesExhausted and releases all memory.
    let tiny_pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(128));
    let oom_ctx = morsel_context(
        4,
        4,
        MorselMode::Required,
        false,
        Some(Arc::clone(&tiny_pool)),
    );
    register_tpch_tables(&oom_ctx, &paths).await?;
    let oom_plan = oom_ctx.sql(sql).await?.create_physical_plan().await?;
    let err = collect(oom_plan, oom_ctx.task_ctx()).await.unwrap_err();
    assert!(
        matches!(err, DataFusionError::ResourcesExhausted(_)),
        "expected ResourcesExhausted, got: {err}"
    );
    timeout(Duration::from_secs(5), async {
        while tiny_pool.reserved() != 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("pool reservation leaked after mid-DAG OOM");

    Ok(())
}

async fn assert_native_parity(
    stock: &SessionContext,
    native: &SessionContext,
    sql: &str,
    ordered: bool,
) -> Result<Arc<dyn datafusion::physical_plan::ExecutionPlan>> {
    let stock_plan = stock.sql(sql).await?.create_physical_plan().await?;
    let expected = collect(Arc::clone(&stock_plan), stock.task_ctx()).await?;
    let native_plan = native.sql(sql).await?.create_physical_plan().await?;
    assert_eq!(
        native_plan.name(),
        "MorselExec",
        "expected MorselExec for `{sql}`, got:\n{}",
        displayable(native_plan.as_ref()).indent(true)
    );
    let actual = collect(Arc::clone(&native_plan), native.task_ctx()).await?;
    assert_eq!(
        native_plan.schema().fields().len(),
        stock_plan.schema().fields().len(),
        "schema length mismatch for `{sql}`"
    );
    for (nf, sf) in native_plan
        .schema()
        .fields()
        .iter()
        .zip(stock_plan.schema().fields().iter())
    {
        assert_eq!(nf.name(), sf.name(), "field name mismatch for `{sql}`");
        assert_eq!(
            nf.data_type(),
            sf.data_type(),
            "field type mismatch for `{sql}`"
        );
    }
    assert_eq!(
        to_rows(&actual, !ordered)?,
        to_rows(&expected, !ordered)?,
        "row parity mismatch for `{sql}`"
    );
    Ok(native_plan)
}

#[tokio::test]
async fn distinct_count_distinct_having_and_nested_aggregate_parity() -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("g", DataType::Utf8, true),
        Field::new("a", DataType::Int64, true),
        Field::new("b", DataType::Utf8, true),
        Field::new("v", DataType::Int64, true),
    ]));
    let batch1 = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec![
                Some("x"),
                Some("x"),
                Some("x"),
                Some("y"),
                None,
                None,
            ])) as ArrayRef,
            Arc::new(Int64Array::from(vec![
                Some(1),
                Some(1),
                Some(2),
                None,
                Some(3),
                Some(3),
            ])),
            Arc::new(StringArray::from(vec![
                Some("p"),
                Some("q"),
                Some("p"),
                Some("r"),
                None,
                Some("s"),
            ])),
            Arc::new(Int64Array::from(vec![
                Some(10),
                Some(20),
                Some(30),
                Some(40),
                Some(50),
                Some(60),
            ])),
        ],
    )?;
    let batch2 = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec![
                Some("x"),
                Some("y"),
                Some("y"),
                Some("z"),
            ])) as ArrayRef,
            Arc::new(Int64Array::from(vec![Some(2), Some(5), Some(5), None])),
            Arc::new(StringArray::from(vec![Some("q"), Some("r"), None, None])),
            Arc::new(Int64Array::from(vec![
                Some(5),
                Some(15),
                Some(25),
                Some(35),
            ])),
        ],
    )?;
    let table = Arc::new(MemTable::try_new(
        Arc::clone(&schema),
        vec![vec![batch1, batch2]],
    )?);
    let empty_table = Arc::new(MemTable::try_new(Arc::clone(&schema), vec![vec![]])?);

    let stock = stock_context(4, 2);
    let native = morsel_context(4, 2, MorselMode::Required, false, None);
    for ctx in [&stock, &native] {
        ctx.register_table("t", table.clone())?;
        ctx.register_table("empty_t", empty_table.clone())?;
    }

    let queries = [
        // SELECT DISTINCT (single-col, multi-col, NULLs, empty, ordered)
        ("SELECT DISTINCT a FROM t", false),
        ("SELECT DISTINCT g, a FROM t", false),
        (
            "SELECT DISTINCT g, b FROM t ORDER BY g ASC NULLS LAST, b ASC NULLS LAST",
            true,
        ),
        ("SELECT DISTINCT a FROM empty_t", false),
        // COUNT(DISTINCT) rewritten via SingleDistinctToGroupBy
        ("SELECT COUNT(DISTINCT a) FROM t", false),
        ("SELECT COUNT(DISTINCT b) FROM t", false),
        ("SELECT COUNT(DISTINCT a) FROM empty_t", false),
        ("SELECT g, COUNT(DISTINCT a) FROM t GROUP BY g", false),
        (
            "SELECT g, COUNT(DISTINCT b) FROM t GROUP BY g ORDER BY g ASC NULLS LAST",
            true,
        ),
        // Direct COUNT(DISTINCT) inside AggregateSink (not rewritten by SingleDistinctToGroupBy)
        ("SELECT COUNT(DISTINCT a), SUM(v) FROM t", false),
        ("SELECT COUNT(DISTINCT a), COUNT(DISTINCT b) FROM t", false),
        (
            "SELECT g, COUNT(DISTINCT a), SUM(v) FROM t GROUP BY g",
            false,
        ),
        (
            "SELECT g, COUNT(DISTINCT a), COUNT(DISTINCT b), AVG(v) FROM t GROUP BY g ORDER BY g ASC NULLS LAST",
            true,
        ),
        (
            "SELECT g, COUNT(DISTINCT a), SUM(v) FROM empty_t GROUP BY g",
            false,
        ),
        (
            "SELECT COUNT(DISTINCT a), COUNT(DISTINCT b) FROM empty_t",
            false,
        ),
        // HAVING on Count, GroupCount, and Aggregate (passing and empty-result)
        ("SELECT COUNT(*) AS c FROM t HAVING COUNT(*) > 0", false),
        ("SELECT COUNT(*) AS c FROM t HAVING COUNT(*) > 1000", false),
        (
            "SELECT COUNT(*) AS c FROM t HAVING COUNT(*) > 1000 ORDER BY c",
            true,
        ),
        (
            "SELECT a, COUNT(*) AS c FROM t GROUP BY a HAVING COUNT(*) > 1",
            false,
        ),
        (
            "SELECT a, COUNT(*) AS c FROM t GROUP BY a HAVING COUNT(*) > 100",
            false,
        ),
        (
            "SELECT g, SUM(v) AS s FROM t GROUP BY g HAVING SUM(v) > 50 ORDER BY s DESC, g ASC NULLS LAST LIMIT 2",
            true,
        ),
        // Nested Aggregate -> Aggregate
        (
            "SELECT cnt, COUNT(*) AS groups FROM (SELECT g, COUNT(*) AS cnt FROM t GROUP BY g) sub GROUP BY cnt ORDER BY cnt",
            true,
        ),
        (
            "SELECT MAX(s), MIN(s) FROM (SELECT g, SUM(v) AS s FROM t GROUP BY g) sub",
            false,
        ),
        // Aggregate as Join input (build, probe, both sides, outer/semi/anti/cross)
        (
            "SELECT l.g, l.s, r.c FROM (SELECT g, SUM(v) AS s FROM t GROUP BY g) l JOIN (SELECT g, COUNT(*) AS c FROM t GROUP BY g) r ON l.g = r.g",
            false,
        ),
        (
            "SELECT t.g, t.v, agg.max_v FROM t LEFT JOIN (SELECT g, MAX(v) AS max_v FROM t GROUP BY g) agg ON t.g = agg.g",
            false,
        ),
        (
            "SELECT t.g, t.v FROM t WHERE t.a IN (SELECT a FROM t GROUP BY a HAVING COUNT(*) >= 2)",
            false,
        ),
    ];

    for (sql, ordered) in queries {
        assert_native_parity(&stock, &native, sql, ordered).await?;
    }

    // Directly exercise LogicalPlan::Distinct(Distinct::All) bypassing ReplaceDistinctWithAggregate.
    let scan_plan = native
        .sql("SELECT g, a FROM t")
        .await?
        .into_optimized_plan()?;
    let distinct_logical = LogicalPlan::Distinct(Distinct::All(Arc::new(scan_plan)));
    let distinct_phys = native
        .state()
        .create_physical_plan(&distinct_logical)
        .await?;
    assert_eq!(distinct_phys.name(), "MorselExec");
    let actual = collect(distinct_phys, native.task_ctx()).await?;
    let expected = stock
        .sql("SELECT DISTINCT g, a FROM t")
        .await?
        .collect()
        .await?;
    assert_eq!(to_rows(&actual, true)?, to_rows(&expected, true)?);

    Ok(())
}

#[tokio::test]
async fn materialized_cte_values_empty_relation_limit_subquery_sort_and_scalar_subqueries()
-> Result<()> {
    let (_dir, paths) = build_tpch_parquet_fixture()?;
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(64 * 1024 * 1024));
    let stock = stock_context(4, 4);
    let native =
        morsel_context(4, 4, MorselMode::Required, false, Some(Arc::clone(&pool)));
    register_tpch_tables(&stock, &paths).await?;
    register_tpch_tables(&native, &paths).await?;
    // `WITH ... AS MATERIALIZED (...)` is parsed under the PostgreSQL dialect.
    stock
        .sql("SET datafusion.sql_parser.dialect = 'PostgreSQL'")
        .await?
        .collect()
        .await?;
    native
        .sql("SET datafusion.sql_parser.dialect = 'PostgreSQL'")
        .await?
        .collect()
        .await?;

    // 1. MaterializedCte referenced by BOTH main query and a ScalarSubquery (Q15 pattern):
    //    CTE body executes exactly once!
    let cte_sql = "
        WITH revenue0 AS MATERIALIZED (
            SELECT
                l_suppkey AS supplier_no,
                SUM(l_extendedprice * (1 - l_discount)) AS total_revenue
            FROM lineitem
            WHERE l_shipdate >= DATE '1996-01-01'
              AND l_shipdate < DATE '1996-04-01'
            GROUP BY l_suppkey
        )
        SELECT s_suppkey, s_name, total_revenue
        FROM supplier
        JOIN revenue0 ON s_suppkey = supplier_no
        WHERE total_revenue = (SELECT MAX(total_revenue) FROM revenue0)
        ORDER BY s_suppkey
    ";
    let cte_plan = assert_native_parity(&stock, &native, cte_sql, true).await?;
    assert!(
        displayable(cte_plan.as_ref())
            .indent(false)
            .to_string()
            .contains("Stage cte_revenue0"),
        "expected MaterializedCte stage in EXPLAIN"
    );
    // 1 aggregate_finalizations for CTE body + 1 for MAX(total_revenue) = 2 (NOT 3!).
    assert_eq!(
        cte_plan
            .metrics()
            .and_then(|m| m.sum_by_name("aggregate_finalizations"))
            .map_or(0, |v| v.as_usize()),
        2,
        "MaterializedCte body should execute only once"
    );
    assert_eq!(pool.reserved(), 0);

    // 2. MaterializedCte self-join (same stage referenced by both probe and build of one Join).
    let self_join_cte = "
        WITH c AS MATERIALIZED (
            SELECT n_regionkey, COUNT(*) AS cnt FROM nation GROUP BY n_regionkey
        )
        SELECT a.n_regionkey, a.cnt, b.cnt AS b_cnt
        FROM c a
        JOIN c b ON a.n_regionkey = b.n_regionkey
        ORDER BY a.n_regionkey
    ";
    assert_native_parity(&stock, &native, self_join_cte, true).await?;
    assert_eq!(pool.reserved(), 0);

    // 3. EmptyRelation (produce_one_row=true and false) and VALUES leaves.
    for (sql, ordered) in [
        ("SELECT 1 AS a, 2 + 3 AS b, 'ok' AS c", true),
        (
            "SELECT * FROM (VALUES (1, 'a'), (2, 'b'), (3, NULL)) AS v(x, y) ORDER BY x DESC",
            true,
        ),
        (
            "SELECT v.x, r.r_name FROM (VALUES (0), (2), (3)) AS v(x) JOIN region r ON v.x = r.r_regionkey ORDER BY v.x",
            true,
        ),
        (
            "SELECT COUNT(*) FROM (VALUES (10), (20), (30)) AS v(x)",
            true,
        ),
        ("SELECT n_name FROM nation WHERE false", true),
    ] {
        assert_native_parity(&stock, &native, sql, ordered).await?;
    }

    // 4. Standalone LIMIT / OFFSET (including LIMIT 0, ordered Sort -> Filter -> Limit, TopN below Join, nested Sort).
    for sql in [
        "SELECT n_nationkey FROM nation LIMIT 3",
        "SELECT n_nationkey FROM nation LIMIT 0",
        "SELECT n_nationkey FROM nation OFFSET 2",
        "SELECT n_nationkey FROM nation LIMIT 3 OFFSET 2",
        "SELECT n_regionkey, COUNT(*) FROM nation GROUP BY n_regionkey LIMIT 2",
    ] {
        let plan = native.sql(sql).await?.create_physical_plan().await?;
        assert_eq!(plan.name(), "MorselExec", "expected MorselExec for `{sql}`");
        let actual = collect(plan, native.task_ctx()).await?;
        let expected = stock.sql(sql).await?.collect().await?;
        let actual_rows: usize = actual.iter().map(RecordBatch::num_rows).sum();
        let expected_rows: usize = expected.iter().map(RecordBatch::num_rows).sum();
        assert_eq!(actual_rows, expected_rows, "row count mismatch for `{sql}`");
    }

    for (sql, ordered) in [
        (
            "SELECT * FROM (SELECT n_nationkey, n_regionkey FROM nation ORDER BY n_nationkey ASC LIMIT 6) sub WHERE n_regionkey >= 1",
            true,
        ),
        (
            "SELECT * FROM (SELECT n_nationkey, n_regionkey FROM nation ORDER BY n_nationkey ASC) sub WHERE n_regionkey >= 1 LIMIT 3 OFFSET 1",
            true,
        ),
        (
            "SELECT a.n_nationkey, r.r_name FROM (SELECT n_nationkey, n_regionkey FROM nation ORDER BY n_nationkey ASC LIMIT 5) a JOIN region r ON a.n_regionkey = r.r_regionkey ORDER BY a.n_nationkey",
            true,
        ),
        (
            "SELECT * FROM (SELECT n_nationkey, n_name FROM nation ORDER BY n_nationkey DESC LIMIT 5) sub ORDER BY n_name ASC LIMIT 3",
            true,
        ),
    ] {
        assert_native_parity(&stock, &native, sql, ordered).await?;
    }

    // 5. Uncorrelated scalar subqueries: 0 rows -> NULL, 1 row -> value, nested scalar subquery,
    //    >1 rows -> execution error, plus sequential & concurrent reuse of the same plan.
    for (sql, ordered) in [
        (
            "SELECT n_nationkey, (SELECT MAX(r_regionkey) FROM region) AS max_r FROM nation ORDER BY n_nationkey",
            true,
        ),
        (
            "SELECT n_nationkey, (SELECT r_regionkey FROM region WHERE false) AS empty_r FROM nation ORDER BY n_nationkey",
            true,
        ),
        (
            "SELECT p_partkey FROM part WHERE p_size > (SELECT AVG(p_size) FROM part WHERE p_size > (SELECT MIN(p_size) FROM part)) ORDER BY p_partkey",
            true,
        ),
    ] {
        let plan = assert_native_parity(&stock, &native, sql, ordered).await?;
        // Sequential re-execution of the exact same Arc<MorselExec>.
        let rerun = collect(Arc::clone(&plan), native.task_ctx()).await?;
        let expected = stock.sql(sql).await?.collect().await?;
        assert_eq!(to_rows(&rerun, !ordered)?, to_rows(&expected, !ordered)?);
        // Concurrent re-execution of the exact same Arc<MorselExec>.
        let (r1, r2) = tokio::try_join!(
            collect(Arc::clone(&plan), native.task_ctx()),
            collect(Arc::clone(&plan), native.task_ctx()),
        )?;
        assert_eq!(to_rows(&r1, !ordered)?, to_rows(&expected, !ordered)?);
        assert_eq!(to_rows(&r2, !ordered)?, to_rows(&expected, !ordered)?);
    }

    // >1 rows in scalar subquery returns execution error and leaks zero memory.
    let multi_row_sq = "SELECT n_nationkey FROM nation WHERE n_regionkey = (SELECT r_regionkey FROM region)";
    let err_plan = native
        .sql(multi_row_sq)
        .await?
        .create_physical_plan()
        .await?;
    assert_eq!(err_plan.name(), "MorselExec");
    let err = collect(err_plan, native.task_ctx()).await.unwrap_err();
    assert!(
        err.to_string().contains("more than one row"),
        "expected scalar subquery cardinality error, got: {err}"
    );
    assert_eq!(pool.reserved(), 0);

    Ok(())
}

#[tokio::test]
async fn still_unsupported_operators_reject_in_required_and_fallback_in_prefer()
-> Result<()> {
    let (_dir, paths) = build_tpch_parquet_fixture()?;
    let stock = stock_context(2, 4);
    let prefer = morsel_context(2, 4, MorselMode::Prefer, false, None);
    let required = morsel_context(2, 4, MorselMode::Required, false, None);
    for ctx in [&stock, &prefer, &required] {
        register_tpch_tables(ctx, &paths).await?;
    }

    for sql in [
        "SELECT n_nationkey, ROW_NUMBER() OVER (ORDER BY n_nationkey) FROM nation",
        "SELECT r_regionkey FROM region UNION ALL SELECT r_regionkey FROM region",
        "SELECT DISTINCT ON (n_regionkey) n_regionkey, n_nationkey FROM nation ORDER BY n_regionkey, n_nationkey",
        "SELECT COUNT(DISTINCT n_regionkey), SUM(DISTINCT n_nationkey) FROM nation",
        "SELECT MEDIAN(p_size) FROM part",
    ] {
        let err = required
            .sql(sql)
            .await?
            .create_physical_plan()
            .await
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("morsel_mode=required"),
            "expected morsel_mode=required rejection for `{sql}`, got: {err}"
        );
        let fallback = prefer.sql(sql).await?.create_physical_plan().await?;
        assert_ne!(
            fallback.name(),
            "MorselExec",
            "expected stock fallback in prefer mode for `{sql}`"
        );
        let actual = collect(fallback, prefer.task_ctx()).await?;
        let expected = stock.sql(sql).await?.collect().await?;
        assert_eq!(to_rows(&actual, true)?, to_rows(&expected, true)?, "{sql}");
    }
    Ok(())
}

#[tokio::test]
async fn limit_streams_without_stage_bounds_memory_and_join_prunes_filter_columns()
-> Result<()> {
    // 1. One 8 MB MemTable batch (1_048_576 Int64 rows) under a 2 MB GreedyMemoryPool.
    let n_rows = 1_048_576usize;
    let big_schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int64, false)]));
    let big_batch = RecordBatch::try_new(
        Arc::clone(&big_schema),
        vec![Arc::new(Int64Array::from_iter_values(0..n_rows as i64)) as ArrayRef],
    )?;
    let big_mem = Arc::new(MemTable::try_new(
        Arc::clone(&big_schema),
        vec![vec![big_batch.clone()]],
    )?);

    // Multi-row-group Parquet table (~16 MB uncompressed: 524_288 rows × 32 bytes/row)
    // tested with LIMIT 500000 under a tight 4 MB pool to prove streaming without buffering.
    let pq_dir = TempDir::new()?;
    let pq_rows = 524_288usize;
    let pq_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("v1", DataType::Int64, false),
        Field::new("v2", DataType::Int64, false),
        Field::new("v3", DataType::Int64, false),
    ]));
    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(16_384))
        .build();
    let pq_path = pq_dir.path().join("0.parquet");
    {
        let mut writer = ArrowWriter::try_new(
            File::create(&pq_path)?,
            Arc::clone(&pq_schema),
            Some(props),
        )?;
        let chunk = 65_536usize;
        for start in (0..pq_rows).step_by(chunk) {
            let end = (start + chunk).min(pq_rows);
            let batch = RecordBatch::try_new(
                Arc::clone(&pq_schema),
                vec![
                    Arc::new(Int64Array::from_iter_values(start as i64..end as i64))
                        as ArrayRef,
                    Arc::new(Int64Array::from_iter_values(start as i64..end as i64)),
                    Arc::new(Int64Array::from_iter_values(start as i64..end as i64)),
                    Arc::new(Int64Array::from_iter_values(start as i64..end as i64)),
                ],
            )?;
            writer.write(&batch)?;
        }
        writer.close()?;
    }

    let pool_2mb: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(2 * 1024 * 1024));
    let ctx_2mb = morsel_context(
        4,
        8192,
        MorselMode::Required,
        false,
        Some(Arc::clone(&pool_2mb)),
    );
    ctx_2mb.register_table("big", big_mem.clone())?;

    // EXPLAIN for scan + LIMIT shows NO `[Stage` (single streaming Sink::Limit).
    for sql in [
        "SELECT k FROM big LIMIT 1",
        "SELECT k FROM big LIMIT 500000",
        "SELECT k + 1 FROM big LIMIT 100 OFFSET 5",
    ] {
        let plan = ctx_2mb.sql(sql).await?.create_physical_plan().await?;
        let display = displayable(plan.as_ref()).indent(false).to_string();
        assert_eq!(plan.name(), "MorselExec");
        assert!(
            !display.contains("[Stage"),
            "scan+LIMIT should stream without staging for `{sql}`, got:\n{display}"
        );
        let batches = collect(plan, ctx_2mb.task_ctx()).await?;
        let got_rows: usize = batches.iter().map(RecordBatch::num_rows).sum();
        let want_rows = if sql.contains("LIMIT 1 ") || sql.ends_with("LIMIT 1") {
            1
        } else if sql.contains("LIMIT 500000") {
            500_000
        } else {
            100
        };
        assert_eq!(got_rows, want_rows, "{sql}");
        assert_eq!(pool_2mb.reserved(), 0, "leaked memory after `{sql}`");
    }

    // LIMIT feeding an Aggregate or Join must create at most 1 stage (never 2) and
    // must not charge pre-existing MemTable buffers against the 2 MB pool.
    let agg_over_limit = "SELECT COUNT(*) FROM (SELECT k FROM big LIMIT 1)";
    let agg_plan = ctx_2mb
        .sql(agg_over_limit)
        .await?
        .create_physical_plan()
        .await?;
    let agg_display = displayable(agg_plan.as_ref()).indent(false).to_string();
    assert_eq!(
        agg_display.matches("[Stage").count(),
        1,
        "LIMIT feeding Aggregate should create only 1 stage:\n{agg_display}"
    );
    let agg_batches = collect(agg_plan, ctx_2mb.task_ctx()).await?;
    let count_val = agg_batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(count_val, 1);
    assert_eq!(pool_2mb.reserved(), 0);

    let big_agg_over_limit = "SELECT COUNT(*) FROM (SELECT k FROM big LIMIT 500000)";
    let big_agg_plan = ctx_2mb
        .sql(big_agg_over_limit)
        .await?
        .create_physical_plan()
        .await?;
    let big_agg_display = displayable(big_agg_plan.as_ref()).indent(false).to_string();
    assert_eq!(
        big_agg_display.matches("[Stage").count(),
        1,
        "large MemTable LIMIT feeding Aggregate should create only 1 stage:\n{big_agg_display}"
    );
    let big_agg_batches = collect(big_agg_plan, ctx_2mb.task_ctx()).await?;
    let big_count_val = big_agg_batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(big_count_val, 500_000);
    assert_eq!(pool_2mb.reserved(), 0);

    // Large Parquet scan + LIMIT 500000 (~16 MB) under a tight 4 MB pool streams
    // cleanly without staging or OOMing.
    let pool_4mb: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(4 * 1024 * 1024));
    let ctx_4mb = morsel_context(
        4,
        8192,
        MorselMode::Required,
        false,
        Some(Arc::clone(&pool_4mb)),
    );
    ctx_4mb
        .register_parquet(
            "pq_big",
            pq_dir.path().to_str().unwrap(),
            ParquetReadOptions::default(),
        )
        .await?;
    let pq_limit_sql = "SELECT * FROM pq_big LIMIT 500000";
    let pq_plan = ctx_4mb
        .sql(pq_limit_sql)
        .await?
        .create_physical_plan()
        .await?;
    let pq_display = displayable(pq_plan.as_ref()).indent(false).to_string();
    assert_eq!(pq_plan.name(), "MorselExec");
    assert!(
        !pq_display.contains("[Stage"),
        "Parquet scan+LIMIT should stream without staging:\n{pq_display}"
    );
    let mut stream = execute_stream(pq_plan, ctx_4mb.task_ctx())?;
    let mut pq_got_rows = 0usize;
    while let Some(batch) = stream.next().await.transpose()? {
        pq_got_rows += batch.num_rows();
    }
    assert_eq!(pq_got_rows, 500_000);
    drop(stream);
    timeout(Duration::from_secs(5), async {
        while pool_4mb.reserved() != 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("pool reservation leaked after large Parquet LIMIT");

    // 2. Join input filter column pruning at workers=4.
    let t_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("g", DataType::Utf8, false),
        Field::new("v", DataType::Int64, false),
    ]));
    let t_batch = RecordBatch::try_new(
        Arc::clone(&t_schema),
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3, 4])) as ArrayRef,
            Arc::new(StringArray::from(vec!["a", "b", "a", "c"])),
            Arc::new(Int64Array::from(vec![10, 20, 30, 40])),
        ],
    )?;
    let d_schema = Arc::new(Schema::new(vec![
        Field::new("dk", DataType::Int64, false),
        Field::new("dname", DataType::Utf8, false),
    ]));
    let d_batch = RecordBatch::try_new(
        Arc::clone(&d_schema),
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3])) as ArrayRef,
            Arc::new(StringArray::from(vec!["d1", "d2", "d3"])),
        ],
    )?;
    let stock = stock_context(4, 8192);
    let native = morsel_context(4, 8192, MorselMode::Required, false, None);
    for ctx in [&stock, &native] {
        ctx.register_table(
            "t",
            Arc::new(MemTable::try_new(
                Arc::clone(&t_schema),
                vec![vec![t_batch.clone()]],
            )?),
        )?;
        ctx.register_table(
            "d",
            Arc::new(MemTable::try_new(
                Arc::clone(&d_schema),
                vec![vec![d_batch.clone()]],
            )?),
        )?;
    }
    let join_sql = "SELECT d.dname, x.k FROM d JOIN (SELECT k, v FROM t WHERE g = 'a') x ON d.dk = x.k";
    let join_plan = assert_native_parity(&stock, &native, join_sql, false).await?;
    let join_display = displayable(join_plan.as_ref()).indent(false).to_string();
    assert!(
        join_display.contains("projection=[0]"),
        "expected filter-only column pruning on join input at workers=4, got:\n{join_display}"
    );
    Ok(())
}
