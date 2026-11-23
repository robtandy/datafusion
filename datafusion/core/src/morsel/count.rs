// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::Arc;

use arrow::array::{BooleanArray, Int64Array, RecordBatch};
use arrow::compute::filter_record_batch;
use arrow::datatypes::SchemaRef;
use arrow::error::ArrowError;
use datafusion_common::cast::as_boolean_array;
use datafusion_common::{Result, ScalarValue, exec_datafusion_err, internal_err};
use datafusion_expr::expr::AggregateFunctionParams;
use datafusion_expr::logical_plan::Aggregate;
use datafusion_expr::utils::COUNT_STAR_EXPANSION;
use datafusion_expr::{ColumnarValue, Expr};
use datafusion_functions_aggregate::count::Count;
use datafusion_physical_expr::PhysicalExpr;

pub(super) fn count_star(aggregate: &Aggregate) -> Option<&Expr> {
    let [expr] = aggregate.aggr_expr.as_slice() else {
        return None;
    };
    let mut unaliased = expr;
    while let Expr::Alias(alias) = unaliased {
        unaliased = &alias.expr;
    }
    let Expr::AggregateFunction(function) = unaliased else {
        return None;
    };
    let AggregateFunctionParams {
        args,
        distinct,
        filter,
        order_by,
        null_treatment,
    } = &function.params;
    (aggregate.group_expr.is_empty()
        && function.func.inner().is::<Count>()
        && matches!(args.as_slice(), [Expr::Literal(value, _)] if value == &COUNT_STAR_EXPANSION)
        && !distinct
        && filter.is_none()
        && order_by.is_empty()
        && null_treatment.is_none())
    .then_some(expr)
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct CountState(i64);

impl CountState {
    pub(super) fn add_rows(&mut self, rows: usize) -> Result<()> {
        let rows = i64::try_from(rows).map_err(|_| count_overflow())?;
        self.combine(Self(rows))
    }

    pub(super) fn combine(&mut self, local: Self) -> Result<()> {
        self.0 = self.0.checked_add(local.0).ok_or_else(count_overflow)?;
        Ok(())
    }
}

fn count_overflow() -> datafusion_common::DataFusionError {
    exec_datafusion_err!("Morsel COUNT(*) exceeds Int64")
}

#[derive(Debug)]
pub(super) struct CountSink {
    schema: SchemaRef,
    #[cfg(test)]
    pub(super) finalize_gate: Option<super::tests::FinalizeGate>,
}

impl CountSink {
    pub(super) fn new(schema: SchemaRef) -> Self {
        Self {
            schema,
            #[cfg(test)]
            finalize_gate: None,
        }
    }

    pub(super) fn finalize(&self, combined: CountState) -> Result<FinalizedCount> {
        Ok(FinalizedCount(RecordBatch::try_new(
            Arc::clone(&self.schema),
            vec![Arc::new(Int64Array::from(vec![combined.0]))],
        )?))
    }
}

pub(super) fn validate_mask_length(
    batch: &RecordBatch,
    mask: &BooleanArray,
) -> Result<()> {
    if mask.len() > batch.num_rows() {
        return Err(ArrowError::InvalidArgumentError(format!(
            "Filter predicate of length {} is larger than target array of length {}",
            mask.len(),
            batch.num_rows()
        ))
        .into());
    }
    Ok(())
}

pub(super) fn count_mask_rows(batch: &RecordBatch, mask: &BooleanArray) -> Result<usize> {
    if mask.len() == batch.num_rows() {
        Ok(mask.true_count())
    } else {
        validate_mask_length(batch, mask)?;
        Ok(filter_record_batch(batch, mask)?.num_rows())
    }
}

pub(super) fn count_filtered_rows(
    batch: &RecordBatch,
    predicate: &Arc<dyn PhysicalExpr>,
) -> Result<usize> {
    match predicate.evaluate(batch)? {
        ColumnarValue::Scalar(ScalarValue::Boolean(Some(true))) => Ok(batch.num_rows()),
        ColumnarValue::Scalar(ScalarValue::Boolean(Some(false) | None)) => Ok(0),
        ColumnarValue::Scalar(_) => {
            internal_err!("Cannot create filter_array from non-boolean predicates")
        }
        ColumnarValue::Array(array) => {
            let Ok(mask) = as_boolean_array(&array) else {
                return internal_err!(
                    "Cannot create filter_array from non-boolean predicates"
                );
            };
            count_mask_rows(batch, mask)
        }
    }
}

pub(super) struct FinalizedCount(RecordBatch);

impl FinalizedCount {
    pub(super) fn into_batch(self) -> RecordBatch {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use std::fmt;

    use arrow::array::{Int32Array, RecordBatchOptions};
    use arrow::buffer::{BooleanBuffer, NullBuffer};
    use arrow::datatypes::{DataType, Field, FieldRef, Schema};
    use datafusion_common::exec_err;

    use super::*;

    #[derive(Debug, PartialEq, Eq, Hash)]
    enum FixedMaskExpr {
        Scalar(ScalarValue),
        IntArray,
        Error,
    }

    impl fmt::Display for FixedMaskExpr {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "FixedMaskExpr({self:?})")
        }
    }

    impl PhysicalExpr for FixedMaskExpr {
        fn fmt_sql(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            fmt::Display::fmt(self, f)
        }

        fn return_field(&self, _schema: &Schema) -> Result<FieldRef> {
            Ok(Arc::new(Field::new("mask", DataType::Boolean, true)))
        }

        fn evaluate(&self, batch: &RecordBatch) -> Result<ColumnarValue> {
            match self {
                Self::Scalar(scalar) => Ok(ColumnarValue::Scalar(scalar.clone())),
                Self::IntArray => {
                    Ok(ColumnarValue::Array(Arc::new(Int32Array::from(vec![
                        1;
                        batch
                            .num_rows(
                            )
                    ]))))
                }
                Self::Error => exec_err!("predicate evaluation failed"),
            }
        }

        fn children(&self) -> Vec<&Arc<dyn PhysicalExpr>> {
            vec![]
        }

        fn with_new_children(
            self: Arc<Self>,
            children: Vec<Arc<dyn PhysicalExpr>>,
        ) -> Result<Arc<dyn PhysicalExpr>> {
            assert!(children.is_empty());
            Ok(self)
        }
    }

    fn int_batch(rows: usize) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, false)]));
        let values: Vec<i32> = (0..rows as i32).collect();
        RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(values))]).unwrap()
    }

    fn zero_col_batch(rows: usize) -> RecordBatch {
        RecordBatch::try_new_with_options(
            Arc::new(Schema::empty()),
            vec![],
            &RecordBatchOptions::new().with_row_count(Some(rows)),
        )
        .unwrap()
    }

    #[test]
    fn count_mask_ignores_nulls_even_with_set_value_bits() -> Result<()> {
        // Underlying value bits are [true, true, false, true], validity is [valid, null, valid, null].
        // Only index 0 is valid AND true.
        let values = BooleanBuffer::from(vec![true, true, false, true]);
        let nulls = NullBuffer::from(vec![true, false, true, false]);
        let mask = BooleanArray::new(values, Some(nulls));
        let batch = int_batch(4);
        assert_eq!(count_mask_rows(&batch, &mask)?, 1);
        assert_eq!(
            count_mask_rows(&batch, &mask)?,
            filter_record_batch(&batch, &mask)?.num_rows()
        );
        Ok(())
    }

    #[test]
    fn count_filtered_rows_handles_scalars_and_empty_batches() -> Result<()> {
        for batch in [
            int_batch(5),
            int_batch(0),
            zero_col_batch(5),
            zero_col_batch(0),
        ] {
            let expected = batch.num_rows();
            for (scalar, want) in [
                (ScalarValue::Boolean(Some(true)), expected),
                (ScalarValue::Boolean(Some(false)), 0),
                (ScalarValue::Boolean(None), 0),
            ] {
                let expr: Arc<dyn PhysicalExpr> = Arc::new(FixedMaskExpr::Scalar(scalar));
                assert_eq!(count_filtered_rows(&batch, &expr)?, want);
            }
            let bad_scalar: Arc<dyn PhysicalExpr> =
                Arc::new(FixedMaskExpr::Scalar(ScalarValue::Int32(Some(1))));
            let err = count_filtered_rows(&batch, &bad_scalar).unwrap_err();
            assert!(err.to_string().contains("non-boolean predicates"), "{err}");
            let bad_array: Arc<dyn PhysicalExpr> = Arc::new(FixedMaskExpr::IntArray);
            let err = count_filtered_rows(&batch, &bad_array).unwrap_err();
            assert!(err.to_string().contains("non-boolean predicates"), "{err}");
            let fail_expr: Arc<dyn PhysicalExpr> = Arc::new(FixedMaskExpr::Error);
            let err = count_filtered_rows(&batch, &fail_expr).unwrap_err();
            assert!(
                err.to_string().contains("predicate evaluation failed"),
                "{err}"
            );
        }
        Ok(())
    }

    #[test]
    fn count_mask_validates_mask_length_for_columned_and_zero_column_batches()
    -> Result<()> {
        for batch in [int_batch(2), zero_col_batch(2)] {
            let exact = BooleanArray::from(vec![Some(true), None]);
            assert_eq!(count_mask_rows(&batch, &exact)?, 1);

            let undersized = BooleanArray::from(vec![Some(true)]);
            assert_eq!(
                count_mask_rows(&batch, &undersized)?,
                filter_record_batch(&batch, &undersized)?.num_rows()
            );

            let oversized = BooleanArray::from(vec![true, true, true]);
            let err = count_mask_rows(&batch, &oversized).unwrap_err();
            assert!(
                err.to_string().contains(
                    "Filter predicate of length 3 is larger than target array of length 2"
                ),
                "{err}"
            );
        }
        Ok(())
    }
}
