// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::cell::RefCell;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use arrow::array::RecordBatch;
use arrow::datatypes::{Schema, SchemaRef};
use bytes::Bytes;
use datafusion_common::instant::Instant;
use datafusion_common::{
    DataFusionError, Result, internal_datafusion_err, resources_datafusion_err,
};
use datafusion_datasource::morsel::{Morsel, MorselPlan, MorselPlanner};
use datafusion_execution::memory_pool::{
    MemoryConsumer, MemoryLimit, MemoryPool, MemoryReservation,
};
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::projection::ProjectionExprs;
use datafusion_physical_plan::metrics::{
    BaselineMetrics, Count, ExecutionPlanMetricsSet, Gauge, Label, MetricBuilder, Time,
};
use futures::{StreamExt, stream};
use object_store::{ObjectStore, path::Path};
use parking_lot::Mutex;
use parquet::DecodeResult;
use parquet::arrow::arrow_reader::metrics::ArrowReaderMetrics;
use parquet::arrow::arrow_reader::{
    ArrowReaderMetadata, ParquetRecordBatchReader, RowSelection, RowSelectionPolicy,
    RowSelector,
};
use parquet::arrow::async_reader::AsyncFileReader;
use parquet::arrow::push_decoder::{
    ParquetPushDecoder, ParquetPushDecoderBuilder, RowGroupSelection,
};
use parquet::basic::Type as PhysicalType;
use parquet::errors::ParquetError;
use parquet::file::metadata::ParquetMetaData;
use parquet::file::page_index::offset_index::PageLocation;
use parquet::schema::types::ColumnDescriptor;
use tokio::sync::{mpsc, oneshot, watch};

use crate::ParquetFileMetrics;
use crate::decoder_projection::DecoderProjection;
use crate::metrics::{ByteProgress, RowFilterSkippedFullyMatchedMetric};
use crate::push_decoder::{RgPlanEntry, RowFilterContext};
use crate::row_filter::PrebuiltRowFilterCandidate;

const DEFAULT_ADAPTIVE_PROBE_RGS_PER_MODE: usize = 2;
const DEFAULT_ADAPTIVE_REPROBE_INTERVAL: usize = 64;
const ADAPTIVE_REPROBE_RGS_PER_MODE: usize = 1;
const MIN_RGS_FOR_ADAPTIVE_PROBE: usize = 12;
const RF_BANDIT_WARMUP: usize = 2;
const RF_BANDIT_EXECUTE_INTERVAL: usize = 16;
const RF_BANDIT_OBSERVE_INTERVAL: usize = 2;
const RF_BANDIT_INITIAL_LIKELINESS: u32 = 100;
const RF_BANDIT_SEED: u64 = 0x517c_c1b7_2722_0a95;

/// Execution mode policy for native morsel per-row-group `RowFilter` vs.
/// `BulkDecode` switching.
///
/// # Public Only for Internal Use:
/// This is not a public API and is for internal use only; see [API policy] for details.
///
/// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AdaptiveRowFilterMode {
    /// Dynamically probe `RowFilter` vs. `BulkDecode` per row group and commit
    /// the lower-cost mode, with periodic re-probing.
    #[default]
    Adaptive,
    /// Always execute surviving row groups in `RowFilter` mode.
    ForceRowFilter,
    /// Always execute surviving row groups in `BulkDecode` mode.
    ForceBulk,
    /// Deterministically alternate `RowFilter` and `BulkDecode` across
    /// successive row groups (for exact parity testing).
    AlternateForTest,
}

/// Configuration for runtime-adaptive per-row-group `RowFilter` vs. `BulkDecode`
/// switching in the native morsel Parquet path.
///
/// Attach via `SessionConfig::with_extension` to override defaults in tests or
/// benchmarks. Only affects the native buffered morsel path; stock `FileStream`
/// never uses it.
///
/// # Public Only for Internal Use:
/// This is not a public API and is for internal use only; see [API policy] for details.
///
/// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct AdaptiveRowFilterConfig {
    /// Mode selection policy (default: [`AdaptiveRowFilterMode::Adaptive`]).
    pub mode: AdaptiveRowFilterMode,
    /// Number of row groups probed per mode before committing (default: 3).
    pub probe_rgs_per_mode: usize,
    /// Re-probe every N committed row groups (`0` disables re-probing; default: 64).
    pub reprobe_interval: usize,
}

impl Default for AdaptiveRowFilterConfig {
    fn default() -> Self {
        Self {
            mode: AdaptiveRowFilterMode::Adaptive,
            probe_rgs_per_mode: DEFAULT_ADAPTIVE_PROBE_RGS_PER_MODE,
            reprobe_interval: DEFAULT_ADAPTIVE_REPROBE_INTERVAL,
        }
    }
}

/// Per-row-group execution mode chosen by [`AdaptiveRowFilterContext`].
///
/// # Public Only for Internal Use:
/// This is not a public API and is for internal use only; see [API policy] for details.
///
/// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RgFilterMode {
    RowFilter,
    BulkDecode,
}

impl RgFilterMode {
    fn other(self) -> Self {
        match self {
            Self::RowFilter => Self::BulkDecode,
            Self::BulkDecode => Self::RowFilter,
        }
    }
}

/// Callback returning `(is_infallible, heuristic_cost)` for a physical
/// predicate expression over a given schema.
pub type ConjunctAnalyzer =
    Arc<dyn Fn(&Arc<dyn PhysicalExpr>, &Schema) -> (bool, u64) + Send + Sync>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ControllerPhase {
    InitialProbe {
        dispatched_rf: usize,
        dispatched_bulk: usize,
    },
    Committed {
        winner: RgFilterMode,
        rgs_since_probe: usize,
        next_reprobe_interval: usize,
    },
    Reprobe {
        current_winner: RgFilterMode,
        dispatched_rf: usize,
        dispatched_bulk: usize,
    },
}

#[derive(Debug, Default)]
struct ProbeWindowStats {
    rf_samples: usize,
    rf_cost_ns: u128,
    rf_rows: u128,
    bulk_samples: usize,
    bulk_cost_ns: u128,
    bulk_rows: u128,
}

impl ProbeWindowStats {
    fn reset(&mut self) {
        *self = Self::default();
    }

    fn rf_cost_per_row(&self) -> f64 {
        self.rf_cost_ns as f64 / self.rf_rows.max(1) as f64
    }

    fn bulk_cost_per_row(&self) -> f64 {
        self.bulk_cost_ns as f64 / self.bulk_rows.max(1) as f64
    }
}

fn scaled_reprobe_interval(base: usize, ratio: f64) -> usize {
    if base == 0 {
        return 0;
    }
    if ratio >= 2.0 {
        base.saturating_mul(4)
    } else if ratio >= 1.5 {
        base.saturating_mul(2)
    } else {
        base
    }
}

struct ControllerState {
    phase: ControllerPhase,
    epoch: u64,
    stats: ProbeWindowStats,
}

struct RowFilterOrderBandit {
    initialized: bool,
    num_candidates: usize,
    /// Incumbent candidate order used during warmup/execute and for concurrent
    /// RGs dispatched while a bounded trial swap is in flight.
    permutation: Vec<usize>,
    /// Trial candidate order during `observe` (only `RF_BANDIT_OBSERVE_INTERVAL`
    /// RGs are dispatched with this order so concurrent file lanes never flood
    /// a bad swap across 16+ row groups).
    trial_permutation: Vec<usize>,
    trial_dispatched: usize,
    swap_likeliness: Vec<u32>,
    disable_permutations: bool,
    epoch: u64,
    iteration_count: usize,
    swap_idx: usize,
    right_random_border: u32,
    runtime_sum: f64,
    prev_mean: f64,
    observe: bool,
    warmup: bool,
    rng_state: u64,
}

impl Default for RowFilterOrderBandit {
    fn default() -> Self {
        Self {
            initialized: false,
            num_candidates: 0,
            permutation: Vec::new(),
            trial_permutation: Vec::new(),
            trial_dispatched: 0,
            swap_likeliness: Vec::new(),
            disable_permutations: true,
            epoch: 1,
            iteration_count: 0,
            swap_idx: 0,
            right_random_border: 0,
            runtime_sum: 0.0,
            prev_mean: 0.0,
            observe: false,
            warmup: true,
            rng_state: RF_BANDIT_SEED,
        }
    }
}

impl RowFilterOrderBandit {
    fn next_random_u32(&mut self, upper_exclusive: u32) -> u32 {
        let mut x = self.rng_state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.rng_state = x;
        let mixed = x.wrapping_mul(0x2545_f491_4f6c_dd1d);
        ((mixed >> 32) as u32) % upper_exclusive
    }

    fn compute_initial_order(
        candidates: &[PrebuiltRowFilterCandidate],
        analyzer: Option<&ConjunctAnalyzer>,
    ) -> (Vec<usize>, bool) {
        let n = candidates.len();
        if n <= 1 {
            return ((0..n).collect(), true);
        }
        let mut all_infallible = true;
        let mut costs = Vec::with_capacity(n);
        for cand in candidates {
            let (cand_infallible, cand_cost) = if let Some(analyze) = analyzer {
                let mut inf = true;
                let mut min_cost = u64::MAX;
                for expr in cand.physical_exprs() {
                    let (e_inf, e_cost) = analyze(expr, cand.projected_schema());
                    inf &= e_inf;
                    min_cost = min_cost.min(e_cost);
                }
                (inf, if min_cost == u64::MAX { 100 } else { min_cost })
            } else {
                (!cand.has_fallible(), cand.required_bytes() as u64)
            };
            all_infallible &= cand_infallible;
            costs.push(cand_cost);
        }
        if !all_infallible {
            // Exact DuckDB safety invariant: preserve left-to-right order
            // whenever any conjunct can throw or is volatile.
            return ((0..n).collect(), true);
        }
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&idx| (costs[idx], candidates[idx].required_bytes(), idx));
        (order, false)
    }

    fn current_order(
        &mut self,
        candidates: &[PrebuiltRowFilterCandidate],
        analyzer: Option<&ConjunctAnalyzer>,
    ) -> (Vec<usize>, Option<u64>) {
        if !self.initialized {
            let (order, disable) = Self::compute_initial_order(candidates, analyzer);
            let n = order.len();
            let pairs = n.saturating_sub(1);
            self.num_candidates = n;
            self.permutation = order;
            self.disable_permutations = disable || n <= 1;
            self.swap_likeliness = vec![RF_BANDIT_INITIAL_LIKELINESS; pairs];
            self.right_random_border =
                (RF_BANDIT_INITIAL_LIKELINESS as usize * pairs) as u32;
            self.initialized = true;
        }
        if candidates.len() != self.num_candidates {
            let (order, _) = Self::compute_initial_order(candidates, analyzer);
            return (order, None);
        }
        if self.disable_permutations {
            return (self.permutation.clone(), None);
        }
        if self.observe {
            if self.trial_dispatched < RF_BANDIT_OBSERVE_INTERVAL {
                self.trial_dispatched += 1;
                return (self.trial_permutation.clone(), Some(self.epoch));
            }
            // Bounded trial RGs already in flight: keep concurrent file lanes
            // on the proven incumbent order until the trial finishes.
            return (self.permutation.clone(), None);
        }
        (self.permutation.clone(), Some(self.epoch))
    }

    fn report(&mut self, epoch: u64, cost_per_row: f64) {
        if self.disable_permutations || epoch != self.epoch {
            return;
        }
        self.iteration_count += 1;
        self.runtime_sum += cost_per_row;

        if self.warmup {
            if self.iteration_count == RF_BANDIT_WARMUP {
                self.iteration_count = 0;
                self.runtime_sum = 0.0;
                self.observe = false;
                self.warmup = false;
            }
            return;
        }

        if self.observe && self.iteration_count == RF_BANDIT_OBSERVE_INTERVAL {
            let trial_mean = self.runtime_sum / self.iteration_count as f64;
            if self.prev_mean - trial_mean <= 0.0 {
                if self.swap_likeliness[self.swap_idx] > 1 {
                    self.swap_likeliness[self.swap_idx] /= 2;
                }
            } else {
                self.permutation = self.trial_permutation.clone();
                self.swap_likeliness[self.swap_idx] = RF_BANDIT_INITIAL_LIKELINESS;
            }
            self.epoch = self.epoch.wrapping_add(1);
            self.observe = false;
            self.trial_dispatched = 0;
            self.iteration_count = 0;
            self.runtime_sum = 0.0;
        } else if !self.observe && self.iteration_count == RF_BANDIT_EXECUTE_INTERVAL {
            self.prev_mean = self.runtime_sum / self.iteration_count as f64;
            let random_number = self.next_random_u32(self.right_random_border);
            self.swap_idx = (random_number / RF_BANDIT_INITIAL_LIKELINESS) as usize;
            let likeliness =
                random_number - RF_BANDIT_INITIAL_LIKELINESS * self.swap_idx as u32;
            if self.swap_likeliness[self.swap_idx] > likeliness {
                self.trial_permutation = self.permutation.clone();
                self.trial_permutation
                    .swap(self.swap_idx, self.swap_idx + 1);
                self.trial_dispatched = 0;
                self.observe = true;
                self.epoch = self.epoch.wrapping_add(1);
            }
            self.iteration_count = 0;
            self.runtime_sum = 0.0;
        }
    }
}

struct AdaptiveMetrics {
    row_filter_rgs: Count,
    bulk_decode_rgs: Count,
    fast_path_bulk_rgs: Count,
    mode_switches: Count,
}

/// Query-wide runtime-adaptive controller shared across all file lanes and
/// workers of one native buffered Parquet scan.
///
/// # Public Only for Internal Use:
/// This is not a public API and is for internal use only; see [API policy] for details.
///
/// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
#[doc(hidden)]
pub struct AdaptiveRowFilterContext {
    config: AdaptiveRowFilterConfig,
    num_files: usize,
    has_dynamic_join_filter: bool,
    /// Optional wider projection `(ProjectionExprs, bulk_schema)` used when
    /// `ParquetScan` narrowed its output schema away from static predicate
    /// columns, so `BulkDecode` row groups still decode predicate columns.
    bulk_projection: Option<(ProjectionExprs, SchemaRef)>,
    analyzer: Option<ConjunctAnalyzer>,
    alt_seq: AtomicUsize,
    controller: Mutex<ControllerState>,
    rf_order: Mutex<RowFilterOrderBandit>,
    metrics: AdaptiveMetrics,
}

impl fmt::Debug for AdaptiveRowFilterContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AdaptiveRowFilterContext")
            .field("config", &self.config)
            .field("num_files", &self.num_files)
            .field("has_dynamic_join_filter", &self.has_dynamic_join_filter)
            .field("has_bulk_projection", &self.bulk_projection.is_some())
            .finish_non_exhaustive()
    }
}

impl AdaptiveRowFilterContext {
    pub fn new(
        config: AdaptiveRowFilterConfig,
        num_files: usize,
        has_dynamic_join_filter: bool,
        bulk_projection: Option<(ProjectionExprs, SchemaRef)>,
        analyzer: Option<ConjunctAnalyzer>,
        metrics_set: &ExecutionPlanMetricsSet,
    ) -> Self {
        let metrics = AdaptiveMetrics {
            row_filter_rgs: MetricBuilder::new(metrics_set)
                .counter("adaptive_row_filter_rgs", 0),
            bulk_decode_rgs: MetricBuilder::new(metrics_set)
                .counter("adaptive_bulk_decode_rgs", 0),
            fast_path_bulk_rgs: MetricBuilder::new(metrics_set)
                .counter("adaptive_fast_path_bulk_rgs", 0),
            mode_switches: MetricBuilder::new(metrics_set)
                .counter("adaptive_mode_switches", 0),
        };
        Self {
            config,
            num_files: num_files.max(1),
            has_dynamic_join_filter,
            bulk_projection,
            analyzer,
            alt_seq: AtomicUsize::new(0),
            controller: Mutex::new(ControllerState {
                phase: ControllerPhase::InitialProbe {
                    dispatched_rf: 0,
                    dispatched_bulk: 0,
                },
                epoch: 1,
                stats: ProbeWindowStats::default(),
            }),
            rf_order: Mutex::new(RowFilterOrderBandit::default()),
            metrics,
        }
    }

    pub(crate) fn bulk_projection(&self) -> Option<&(ProjectionExprs, SchemaRef)> {
        self.bulk_projection.as_ref()
    }

    pub(crate) fn has_dynamic_join_filter(&self) -> bool {
        self.has_dynamic_join_filter
    }

    fn assign_rg(
        self: &Arc<Self>,
        can_skip_columns: bool,
        file_surviving_rgs: usize,
        input_rows: u64,
    ) -> RgProbeHandle {
        let (mode, track_mode_epoch) = match self.config.mode {
            AdaptiveRowFilterMode::ForceRowFilter => {
                self.metrics.row_filter_rgs.add(1);
                (RgFilterMode::RowFilter, None)
            }
            AdaptiveRowFilterMode::ForceBulk => {
                self.metrics.bulk_decode_rgs.add(1);
                (RgFilterMode::BulkDecode, None)
            }
            AdaptiveRowFilterMode::AlternateForTest => {
                let seq = self.alt_seq.fetch_add(1, Ordering::Relaxed);
                if seq.is_multiple_of(2) {
                    self.metrics.row_filter_rgs.add(1);
                    (RgFilterMode::RowFilter, None)
                } else {
                    self.metrics.bulk_decode_rgs.add(1);
                    (RgFilterMode::BulkDecode, None)
                }
            }
            AdaptiveRowFilterMode::Adaptive => {
                let probe_target = self.config.probe_rgs_per_mode.max(1);
                let min_rgs_for_probe = probe_target
                    .saturating_mul(4)
                    .max(MIN_RGS_FOR_ADAPTIVE_PROBE);
                let estimated_scan_rgs =
                    file_surviving_rgs.saturating_mul(self.num_files);
                if input_rows == 0 {
                    self.metrics.bulk_decode_rgs.add(1);
                    (RgFilterMode::BulkDecode, None)
                } else if !can_skip_columns {
                    // Fast-path metadata gate: zero non-predicate payload columns,
                    // <= 1 predicate candidate, and no dynamic join filter ->
                    // RowFilter cannot skip any column; BulkDecode is strictly
                    // dominant without probing.
                    self.metrics.fast_path_bulk_rgs.add(1);
                    self.metrics.bulk_decode_rgs.add(1);
                    (RgFilterMode::BulkDecode, None)
                } else if estimated_scan_rgs < min_rgs_for_probe {
                    // Tiny scan with too few row groups to amortize A/B probing:
                    // execute directly in RowFilter mode.
                    self.metrics.row_filter_rgs.add(1);
                    (RgFilterMode::RowFilter, None)
                } else {
                    let mut guard = self.controller.lock();
                    let ControllerState {
                        phase,
                        epoch: state_epoch,
                        stats,
                    } = &mut *guard;
                    let (chosen, epoch) = match phase {
                        ControllerPhase::InitialProbe {
                            dispatched_rf,
                            dispatched_bulk,
                        } => {
                            if *dispatched_rf < probe_target
                                || *dispatched_bulk < probe_target
                            {
                                let mode = if *dispatched_rf <= *dispatched_bulk {
                                    *dispatched_rf += 1;
                                    RgFilterMode::RowFilter
                                } else {
                                    *dispatched_bulk += 1;
                                    RgFilterMode::BulkDecode
                                };
                                (mode, Some(*state_epoch))
                            } else {
                                // All probe RGs are already in flight. Never
                                // over-dispatch slow BulkDecode probes while
                                // waiting: prefer whichever mode has completed
                                // samples faster (or RowFilter as the lower-
                                // regret default while wave 1 finishes).
                                let mode = if stats.rf_samples > 0
                                    && stats.bulk_samples > 0
                                {
                                    if stats.bulk_cost_per_row() < stats.rf_cost_per_row()
                                    {
                                        RgFilterMode::BulkDecode
                                    } else {
                                        RgFilterMode::RowFilter
                                    }
                                } else if stats.bulk_samples > 0 && stats.rf_samples == 0
                                {
                                    RgFilterMode::BulkDecode
                                } else if (stats.rf_samples > 0
                                    && stats.bulk_samples == 0)
                                    || self.has_dynamic_join_filter
                                {
                                    RgFilterMode::RowFilter
                                } else {
                                    RgFilterMode::BulkDecode
                                };
                                (mode, None)
                            }
                        }
                        ControllerPhase::Committed {
                            winner,
                            rgs_since_probe,
                            next_reprobe_interval,
                        } => {
                            let current = *winner;
                            if *next_reprobe_interval > 0
                                && *rgs_since_probe >= *next_reprobe_interval
                            {
                                let challenger = current.other();
                                *state_epoch = state_epoch.wrapping_add(1);
                                stats.reset();
                                let (dispatched_rf, dispatched_bulk) = match challenger {
                                    RgFilterMode::RowFilter => (1, 0),
                                    RgFilterMode::BulkDecode => (0, 1),
                                };
                                *phase = ControllerPhase::Reprobe {
                                    current_winner: current,
                                    dispatched_rf,
                                    dispatched_bulk,
                                };
                                (challenger, Some(*state_epoch))
                            } else {
                                *rgs_since_probe += 1;
                                (current, None)
                            }
                        }
                        ControllerPhase::Reprobe {
                            current_winner,
                            dispatched_rf,
                            dispatched_bulk,
                        } => {
                            let current = *current_winner;
                            let challenger = current.other();
                            if *dispatched_rf < ADAPTIVE_REPROBE_RGS_PER_MODE
                                || *dispatched_bulk < ADAPTIVE_REPROBE_RGS_PER_MODE
                            {
                                let (chal_cnt, cur_cnt) = match challenger {
                                    RgFilterMode::RowFilter => {
                                        (dispatched_rf, dispatched_bulk)
                                    }
                                    RgFilterMode::BulkDecode => {
                                        (dispatched_bulk, dispatched_rf)
                                    }
                                };
                                let mode = if *chal_cnt < ADAPTIVE_REPROBE_RGS_PER_MODE
                                    && *chal_cnt <= *cur_cnt
                                {
                                    *chal_cnt += 1;
                                    challenger
                                } else {
                                    *cur_cnt += 1;
                                    current
                                };
                                (mode, Some(*state_epoch))
                            } else {
                                (current, None)
                            }
                        }
                    };
                    drop(guard);
                    match chosen {
                        RgFilterMode::RowFilter => self.metrics.row_filter_rgs.add(1),
                        RgFilterMode::BulkDecode => self.metrics.bulk_decode_rgs.add(1),
                    }
                    (chosen, epoch)
                }
            }
        };

        RgProbeHandle {
            controller: Arc::clone(self),
            mode,
            input_rows,
            cpu_ns: 0,
            bytes_fetched: 0,
            track_mode_epoch,
            rf_order_epoch: None,
            finished: false,
        }
    }

    fn row_filter_order(
        &self,
        candidates: &[PrebuiltRowFilterCandidate],
    ) -> (Vec<usize>, Option<u64>) {
        self.rf_order
            .lock()
            .current_order(candidates, self.analyzer.as_ref())
    }

    fn report_mode_sample(
        &self,
        mode: RgFilterMode,
        epoch: u64,
        input_rows: u64,
        total_cpu_ns: u64,
        bytes_fetched: u64,
    ) {
        if input_rows == 0 {
            return;
        }
        // Account for both worker CPU (`decode + process + RowFilter`) and
        // compressed I/O bytes fetched (`0.5 ns/byte`), so page-index / column
        // skipping gets credited cleanly without distorting warm-cache CPU.
        let cost_ns = (total_cpu_ns as u128).saturating_add((bytes_fetched as u128) / 2);
        let probe_target = self.config.probe_rgs_per_mode.max(1);
        let mut state = self.controller.lock();
        if epoch != state.epoch {
            return;
        }
        match mode {
            RgFilterMode::RowFilter => {
                state.stats.rf_samples += 1;
                state.stats.rf_cost_ns = state.stats.rf_cost_ns.saturating_add(cost_ns);
                state.stats.rf_rows =
                    state.stats.rf_rows.saturating_add(input_rows as u128);
            }
            RgFilterMode::BulkDecode => {
                state.stats.bulk_samples += 1;
                state.stats.bulk_cost_ns =
                    state.stats.bulk_cost_ns.saturating_add(cost_ns);
                state.stats.bulk_rows =
                    state.stats.bulk_rows.saturating_add(input_rows as u128);
            }
        }
        match state.phase {
            ControllerPhase::InitialProbe { .. } => {
                if state.stats.rf_samples >= probe_target
                    && state.stats.bulk_samples >= probe_target
                {
                    let rf_cost = state.stats.rf_cost_per_row();
                    let bulk_cost = state.stats.bulk_cost_per_row();
                    let (winner, ratio) = if bulk_cost < rf_cost {
                        (RgFilterMode::BulkDecode, rf_cost / bulk_cost.max(1e-9))
                    } else {
                        (RgFilterMode::RowFilter, bulk_cost / rf_cost.max(1e-9))
                    };
                    let next_reprobe_interval =
                        scaled_reprobe_interval(self.config.reprobe_interval, ratio);
                    state.phase = ControllerPhase::Committed {
                        winner,
                        rgs_since_probe: 0,
                        next_reprobe_interval,
                    };
                    self.metrics.mode_switches.add(1);
                }
            }
            ControllerPhase::Reprobe { current_winner, .. } => {
                if state.stats.rf_samples >= ADAPTIVE_REPROBE_RGS_PER_MODE
                    && state.stats.bulk_samples >= ADAPTIVE_REPROBE_RGS_PER_MODE
                {
                    let rf_cost = state.stats.rf_cost_per_row();
                    let bulk_cost = state.stats.bulk_cost_per_row();
                    // 5% hysteresis margin avoids thrashing on measurement noise.
                    let new_winner = match current_winner {
                        RgFilterMode::RowFilter => {
                            if bulk_cost < rf_cost * 0.95 {
                                RgFilterMode::BulkDecode
                            } else {
                                RgFilterMode::RowFilter
                            }
                        }
                        RgFilterMode::BulkDecode => {
                            if rf_cost < bulk_cost * 0.95 {
                                RgFilterMode::RowFilter
                            } else {
                                RgFilterMode::BulkDecode
                            }
                        }
                    };
                    let ratio = match new_winner {
                        RgFilterMode::BulkDecode => rf_cost / bulk_cost.max(1e-9),
                        RgFilterMode::RowFilter => bulk_cost / rf_cost.max(1e-9),
                    };
                    if new_winner != current_winner {
                        self.metrics.mode_switches.add(1);
                    }
                    let next_reprobe_interval =
                        scaled_reprobe_interval(self.config.reprobe_interval, ratio);
                    state.phase = ControllerPhase::Committed {
                        winner: new_winner,
                        rgs_since_probe: 0,
                        next_reprobe_interval,
                    };
                }
            }
            ControllerPhase::Committed { .. } => {}
        }
    }

    fn report_rf_order_sample(
        &self,
        order_epoch: u64,
        input_rows: u64,
        total_cpu_ns: u64,
        bytes_fetched: u64,
    ) {
        if input_rows == 0 {
            return;
        }
        let cost_ns = (total_cpu_ns as u128).saturating_add((bytes_fetched as u128) / 2);
        let cost_per_row = cost_ns as f64 / input_rows as f64;
        self.rf_order.lock().report(order_epoch, cost_per_row);
    }
}

/// Per-row-group telemetry and mode handle carried from file-lane planning
/// through `FilteredRowGroupPlanner` / `BufferedReader` to the pipeline worker.
///
/// # Public Only for Internal Use:
/// This is not a public API and is for internal use only; see [API policy] for details.
///
/// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
#[doc(hidden)]
pub struct RgProbeHandle {
    controller: Arc<AdaptiveRowFilterContext>,
    mode: RgFilterMode,
    input_rows: u64,
    cpu_ns: u64,
    bytes_fetched: u64,
    track_mode_epoch: Option<u64>,
    rf_order_epoch: Option<u64>,
    finished: bool,
}

impl RgProbeHandle {
    pub fn mode(&self) -> RgFilterMode {
        self.mode
    }

    pub fn is_bulk(&self) -> bool {
        self.mode == RgFilterMode::BulkDecode
    }

    pub fn needs_worker_timing(&self) -> bool {
        self.track_mode_epoch.is_some() || self.rf_order_epoch.is_some()
    }

    fn add_cpu_ns(&mut self, ns: u64) {
        self.cpu_ns = self.cpu_ns.saturating_add(ns);
    }

    fn add_bytes_fetched(&mut self, bytes: u64) {
        self.bytes_fetched = self.bytes_fetched.saturating_add(bytes);
    }

    fn set_rf_order_epoch(&mut self, epoch: Option<u64>) {
        self.rf_order_epoch = epoch;
    }

    /// Record final worker CPU (`decode + process`) for this row group and
    /// report completed telemetry to [`AdaptiveRowFilterContext`].
    pub fn finish(mut self, worker_cpu_ns: u64) {
        if self.finished {
            return;
        }
        self.finished = true;
        let total_cpu_ns = self.cpu_ns.saturating_add(worker_cpu_ns);
        if let Some(epoch) = self.track_mode_epoch {
            self.controller.report_mode_sample(
                self.mode,
                epoch,
                self.input_rows,
                total_cpu_ns,
                self.bytes_fetched,
            );
        }
        if let Some(order_epoch) = self.rf_order_epoch {
            self.controller.report_rf_order_sample(
                order_epoch,
                self.input_rows,
                total_cpu_ns,
                self.bytes_fetched,
            );
        }
    }
}

struct CaptureSlot {
    active: bool,
    handle: Option<RgProbeHandle>,
}

thread_local! {
    static INTO_STREAM_CAPTURE: RefCell<CaptureSlot> = const {
        RefCell::new(CaptureSlot {
            active: false,
            handle: None,
        })
    };
}

struct CaptureGuard;

impl Drop for CaptureGuard {
    fn drop(&mut self) {
        INTO_STREAM_CAPTURE.with(|cell| {
            let mut slot = cell.borrow_mut();
            slot.active = false;
            slot.handle = None;
        });
    }
}

fn publish_into_stream_handle(handle: RgProbeHandle) {
    INTO_STREAM_CAPTURE.with(|cell| {
        let mut slot = cell.borrow_mut();
        if slot.active {
            slot.handle = Some(handle);
        }
    });
}

/// Synchronously capture the [`RgProbeHandle`] (if any) emitted by
/// `BufferedReader::into_stream` during `f()`.
///
/// # Public Only for Internal Use:
/// This is not a public API and is for internal use only; see [API policy] for details.
///
/// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
#[doc(hidden)]
pub fn capture_rg_handle<R>(f: impl FnOnce() -> R) -> (R, Option<RgProbeHandle>) {
    INTO_STREAM_CAPTURE.with(|cell| {
        let mut slot = cell.borrow_mut();
        slot.active = true;
        slot.handle = None;
    });
    let _guard = CaptureGuard;
    let res = f();
    let handle = INTO_STREAM_CAPTURE.with(|cell| cell.borrow_mut().handle.take());
    (res, handle)
}

const DEFAULT_MIN_ROW_GROUP_ROWS: usize = 1 << 20;
const DEFAULT_TARGET_SLICE_ROWS: usize = 1 << 17;
const DEFAULT_SPLIT_BUFFER_CAP_BYTES: usize = 256 << 20;
const DEFAULT_MAX_SLICE_COMPRESSED_BYTES: usize = 16 << 20;
const DEFAULT_MAX_COALESCED_REQUEST_BYTES: usize = 1 << 20;
const DEFAULT_MAX_DECODED_BYTES: usize = 128 << 20;
const DEFAULT_MAX_DECODE_AMPLIFICATION: f64 = 2.0;

/// Configuration for experimental intra-row-group Parquet morsel splitting.
///
/// Attach via `SessionConfig::with_extension` to override defaults in tests or
/// benchmarks. Only affects the native buffered Parquet morselizer; stock
/// `FileStream` never uses it.
///
/// # Public Only for Internal Use:
/// This is not a public API and is for internal use only; see [API policy] for details.
///
/// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct ParquetSplitConfig {
    /// Enable intra-row-group splitting for eligible large row groups.
    pub enabled: bool,
    /// Minimum row-group row count to consider splitting (default: 1 Mi rows).
    pub min_row_group_rows: usize,
    /// Target rows per sub-row-group slice (default: 128 Ki rows).
    pub target_slice_rows: usize,
    /// Query-wide compressed split-buffer byte cap across in-flight I/O,
    /// brokers, ready readers, and claimed workers (default: 256 MiB).
    pub split_buffer_cap_bytes: usize,
    /// Per-slice compressed byte cap, including dictionary and shared pages
    /// (default: 16 MiB).
    pub max_slice_compressed_bytes: usize,
    /// Maximum coalesced range request size (default: 1 MiB). A single page
    /// larger than this (within `max_slice_compressed_bytes`) is fetched as-is.
    pub max_coalesced_request_bytes: usize,
    /// Conservative per-slice decoded workspace ceiling (default: 128 MiB).
    pub max_decoded_bytes: usize,
    /// Maximum ratio of summed per-slice requested bytes to unique requested
    /// bytes before rejecting a row group (default: 2.0).
    pub max_decode_amplification: f64,
}

impl Default for ParquetSplitConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            min_row_group_rows: DEFAULT_MIN_ROW_GROUP_ROWS,
            target_slice_rows: DEFAULT_TARGET_SLICE_ROWS,
            split_buffer_cap_bytes: DEFAULT_SPLIT_BUFFER_CAP_BYTES,
            max_slice_compressed_bytes: DEFAULT_MAX_SLICE_COMPRESSED_BYTES,
            max_coalesced_request_bytes: DEFAULT_MAX_COALESCED_REQUEST_BYTES,
            max_decoded_bytes: DEFAULT_MAX_DECODED_BYTES,
            max_decode_amplification: DEFAULT_MAX_DECODE_AMPLIFICATION,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum UnsplitReason {
    Disabled,
    PageIndexDisabled,
    BelowThreshold,
    MetadataOnly,
    UnsupportedColumnType,
    MissingOffsetIndex,
    InvalidOffsetIndex,
    SingleSlice,
    DecodeAmplification,
    SliceCompressedCap,
    BufferCap,
    DecodedWorkspaceCap,
    MemoryPoolCap,
}

impl UnsplitReason {
    fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::PageIndexDisabled => "page_index_disabled",
            Self::BelowThreshold => "below_threshold",
            Self::MetadataOnly => "metadata_only",
            Self::UnsupportedColumnType => "unsupported_column_type",
            Self::MissingOffsetIndex => "missing_offset_index",
            Self::InvalidOffsetIndex => "invalid_offset_index",
            Self::SingleSlice => "single_slice",
            Self::DecodeAmplification => "decode_amplification",
            Self::SliceCompressedCap => "slice_compressed_cap",
            Self::BufferCap => "buffer_cap",
            Self::DecodedWorkspaceCap => "decoded_workspace_cap",
            Self::MemoryPoolCap => "memory_pool_cap",
        }
    }
}

/// Lazily-registered execution metrics for intra-row-group splitting.
struct SplitMetrics {
    metrics: ExecutionPlanMetricsSet,
    inner: Mutex<SplitMetricsInner>,
}

#[derive(Default)]
struct SplitMetricsInner {
    split_row_groups: Option<Count>,
    unsplit_row_groups: HashMap<UnsplitReason, Count>,
    slices: Option<Count>,
    selected_rows: Option<Count>,
    storage_requests: Option<Count>,
    data_bytes_returned: Option<Count>,
    reused_bytes: Option<Count>,
    reservation_wait: Option<Time>,
    peak_unique_compressed_bytes: Option<Gauge>,
}

impl SplitMetrics {
    fn new(metrics: ExecutionPlanMetricsSet) -> Self {
        Self {
            metrics,
            inner: Mutex::new(SplitMetricsInner::default()),
        }
    }

    fn record_split_rg(&self) {
        let mut inner = self.inner.lock();
        let counter = inner.split_row_groups.get_or_insert_with(|| {
            MetricBuilder::new(&self.metrics).counter("parquet_split_row_groups", 0)
        });
        counter.add(1);
    }

    fn record_unsplit_rg(&self, reason: UnsplitReason) {
        let mut inner = self.inner.lock();
        let counter = inner.unsplit_row_groups.entry(reason).or_insert_with(|| {
            MetricBuilder::new(&self.metrics)
                .with_label(Label::new("reason", reason.as_str()))
                .counter("parquet_unsplit_row_groups", 0)
        });
        counter.add(1);
    }

    fn record_slice(&self, selected_rows: usize) {
        let mut inner = self.inner.lock();
        let slices = inner.slices.get_or_insert_with(|| {
            MetricBuilder::new(&self.metrics).counter("parquet_split_slices", 0)
        });
        slices.add(1);
        let rows = inner.selected_rows.get_or_insert_with(|| {
            MetricBuilder::new(&self.metrics).counter("parquet_split_selected_rows", 0)
        });
        rows.add(selected_rows);
    }

    fn record_storage_fetch(&self, requests: usize, bytes: usize) {
        let mut inner = self.inner.lock();
        let req = inner.storage_requests.get_or_insert_with(|| {
            MetricBuilder::new(&self.metrics).counter("parquet_split_storage_requests", 0)
        });
        req.add(requests);
        let returned = inner.data_bytes_returned.get_or_insert_with(|| {
            MetricBuilder::new(&self.metrics)
                .counter("parquet_split_data_bytes_returned", 0)
        });
        returned.add(bytes);
    }

    fn record_reused_bytes(&self, bytes: usize) {
        if bytes == 0 {
            return;
        }
        let mut inner = self.inner.lock();
        let reused = inner.reused_bytes.get_or_insert_with(|| {
            MetricBuilder::new(&self.metrics).counter("parquet_split_reused_bytes", 0)
        });
        reused.add(bytes);
    }

    fn record_reservation_wait(&self, duration: Duration) {
        if duration.is_zero() {
            return;
        }
        let mut inner = self.inner.lock();
        let wait = inner.reservation_wait.get_or_insert_with(|| {
            MetricBuilder::new(&self.metrics)
                .subset_time("parquet_split_reservation_wait", 0)
        });
        wait.add_duration_exact(duration);
    }

    fn update_peak_unique_bytes(&self, peak: usize) {
        let mut inner = self.inner.lock();
        let gauge = inner.peak_unique_compressed_bytes.get_or_insert_with(|| {
            MetricBuilder::new(&self.metrics)
                .bytes_gauge("parquet_split_peak_unique_compressed_bytes", 0)
        });
        gauge.set(peak);
    }
}

/// Query-wide splitting configuration, budget, and metrics shared across all
/// file lanes of one buffered Parquet scan.
pub(crate) struct SplitContext {
    pub(super) config: ParquetSplitConfig,
    budget: Arc<SplitBudget>,
    metrics: Arc<SplitMetrics>,
}

impl SplitContext {
    pub(crate) fn new(
        config: ParquetSplitConfig,
        memory_pool: Option<Arc<dyn MemoryPool>>,
        metrics: ExecutionPlanMetricsSet,
    ) -> Self {
        let metrics = Arc::new(SplitMetrics::new(metrics));
        let budget = Arc::new(SplitBudget::new(
            config.split_buffer_cap_bytes,
            memory_pool,
            Arc::clone(&metrics),
        ));
        Self {
            config,
            budget,
            metrics,
        }
    }

    pub(super) fn is_enabled(&self) -> bool {
        self.config.enabled && self.config.target_slice_rows > 0
    }

    /// Cheap metadata-only check used by `should_load_page_index` to decide
    /// whether loading Offset Indexes solely for splitting is worthwhile.
    pub(super) fn has_index_load_candidate(
        &self,
        metadata: &ParquetMetaData,
        rg_indexes: impl Iterator<Item = usize>,
        reads_leaf: impl Fn(usize) -> bool,
    ) -> bool {
        if !self.is_enabled() {
            return false;
        }
        let schema_descr = metadata.file_metadata().schema_descr();
        let mut any_leaf = false;
        for leaf_idx in 0..schema_descr.num_columns() {
            if !reads_leaf(leaf_idx) {
                continue;
            }
            any_leaf = true;
            if is_flat_fixed_width_leaf(&schema_descr.column(leaf_idx)).is_none() {
                return false;
            }
        }
        if !any_leaf {
            return false;
        }
        for rg_idx in rg_indexes {
            let rg = metadata.row_group(rg_idx);
            let Ok(num_rows) = usize::try_from(rg.num_rows()) else {
                continue;
            };
            if num_rows < self.config.min_row_group_rows
                || num_rows <= self.config.target_slice_rows
            {
                continue;
            }
            let all_have_offset = (0..schema_descr.num_columns()).all(|leaf_idx| {
                !reads_leaf(leaf_idx)
                    || rg.column(leaf_idx).offset_index_offset().is_some()
            });
            if all_have_offset {
                return true;
            }
        }
        false
    }
}

struct SplitBudgetState {
    cap: usize,
    pool_limit: Option<usize>,
    live_bytes: usize,
    peak_live_bytes: usize,
    admitted_frontier_bytes: usize,
    admitted_pool_frontier_bytes: usize,
    active_split_lanes: usize,
    waiting_for_memory: usize,
    in_flight_decode_readers: usize,
    in_flight_filter_planners: usize,
    generation: u64,
}

struct SplitBudget {
    state: Mutex<SplitBudgetState>,
    memory_pool: Option<Arc<dyn MemoryPool>>,
    metrics: Arc<SplitMetrics>,
    changed_tx: watch::Sender<u64>,
    changed_rx: watch::Receiver<u64>,
}

impl SplitBudget {
    fn new(
        cap: usize,
        memory_pool: Option<Arc<dyn MemoryPool>>,
        metrics: Arc<SplitMetrics>,
    ) -> Self {
        let pool_limit = memory_pool.as_ref().and_then(|p| match p.memory_limit() {
            MemoryLimit::Finite(limit) => Some(limit),
            MemoryLimit::Infinite | MemoryLimit::Unknown => None,
        });
        let (changed_tx, changed_rx) = watch::channel(0u64);
        Self {
            state: Mutex::new(SplitBudgetState {
                cap,
                pool_limit,
                live_bytes: 0,
                peak_live_bytes: 0,
                admitted_frontier_bytes: 0,
                admitted_pool_frontier_bytes: 0,
                active_split_lanes: 0,
                waiting_for_memory: 0,
                in_flight_decode_readers: 0,
                in_flight_filter_planners: 0,
                generation: 0,
            }),
            memory_pool,
            metrics,
            changed_tx,
            changed_rx,
        }
    }

    fn pool_limit(&self) -> Option<usize> {
        self.state.lock().pool_limit
    }

    fn signal_change(&self, state: &mut SplitBudgetState) {
        state.generation = state.generation.wrapping_add(1);
        let _ = self.changed_tx.send(state.generation);
    }

    fn frontier_fits(
        state: &SplitBudgetState,
        frontier_bytes: usize,
        pool_frontier_bytes: usize,
    ) -> bool {
        let cap_fits = state.admitted_frontier_bytes.saturating_add(frontier_bytes)
            <= state.cap
            || (state.active_split_lanes == 0 && frontier_bytes <= state.cap);
        let pool_fits = match state.pool_limit {
            Some(limit) => {
                state
                    .admitted_pool_frontier_bytes
                    .saturating_add(pool_frontier_bytes)
                    <= limit
                    || (state.active_split_lanes == 0 && pool_frontier_bytes <= limit)
            }
            None => true,
        };
        cap_fits && pool_fits
    }

    fn try_acquire_frontier(
        self: &Arc<Self>,
        frontier_bytes: usize,
        pool_frontier_bytes: usize,
    ) -> Option<FrontierGuard> {
        let mut state = self.state.lock();
        if Self::frontier_fits(&state, frontier_bytes, pool_frontier_bytes) {
            state.admitted_frontier_bytes += frontier_bytes;
            state.admitted_pool_frontier_bytes += pool_frontier_bytes;
            state.active_split_lanes += 1;
            Some(FrontierGuard {
                budget: Arc::clone(self),
                frontier_bytes,
                pool_frontier_bytes,
            })
        } else {
            None
        }
    }

    async fn acquire_frontier(
        self: &Arc<Self>,
        frontier_bytes: usize,
        pool_frontier_bytes: usize,
    ) -> Result<FrontierGuard> {
        let mut rx = self.changed_rx.clone();
        let mut wait_start: Option<Instant> = None;
        loop {
            let seen = {
                let mut state = self.state.lock();
                if frontier_bytes > state.cap {
                    return Err(resources_datafusion_err!(
                        "Parquet split frontier ({frontier_bytes} bytes) exceeds split buffer cap ({} bytes)",
                        state.cap
                    ));
                }
                if let Some(limit) = state.pool_limit
                    && pool_frontier_bytes > limit
                {
                    return Err(resources_datafusion_err!(
                        "Parquet split pool frontier ({pool_frontier_bytes} bytes) exceeds memory pool limit ({limit} bytes)"
                    ));
                }
                if Self::frontier_fits(&state, frontier_bytes, pool_frontier_bytes) {
                    state.admitted_frontier_bytes += frontier_bytes;
                    state.admitted_pool_frontier_bytes += pool_frontier_bytes;
                    state.active_split_lanes += 1;
                    if let Some(start) = wait_start {
                        self.metrics.record_reservation_wait(start.elapsed());
                    }
                    return Ok(FrontierGuard {
                        budget: Arc::clone(self),
                        frontier_bytes,
                        pool_frontier_bytes,
                    });
                }
                state.generation
            };
            wait_start.get_or_insert_with(Instant::now);
            let _ = rx.wait_for(|&g| g != seen).await;
        }
    }

    /// Reserve compressed bytes for Phase-1 (`coalesced_lengths`), pre-reserved
    /// Phase-2 headroom (`phase2_budget_fn`), and decoded workspace bytes
    /// atomically before issuing a file-lane read.
    async fn reserve_for_fetch(
        self: &Arc<Self>,
        coalesced_lengths: &[usize],
        phase2_budget_fn: impl Fn() -> usize,
        decoded_estimate: usize,
    ) -> Result<(Vec<SplitBufferLease>, Phase2BudgetLease, SplitDecodedLease)> {
        let phase1_compressed: usize = coalesced_lengths.iter().copied().sum();
        let mut rx = self.changed_rx.clone();
        let mut wait_start: Option<Instant> = None;
        loop {
            // Compute Phase-2 headroom outside `self.state` lock to prevent
            // lock-order inversion with `AsyncRangeBroker` dropping `TrackedBytes`.
            let phase2_needed = phase2_budget_fn();
            let compressed_needed = phase1_compressed.saturating_add(phase2_needed);
            let wait_guard;
            let seen = {
                let mut state = self.state.lock();
                let cap_ok =
                    state.live_bytes.saturating_add(compressed_needed) <= state.cap;
                if cap_ok {
                    match self.try_reserve_pool_pair(compressed_needed, decoded_estimate)
                    {
                        Ok((mut compressed_res, decoded_res)) => {
                            state.live_bytes += compressed_needed;
                            let new_peak = if state.live_bytes > state.peak_live_bytes {
                                state.peak_live_bytes = state.live_bytes;
                                Some(state.peak_live_bytes)
                            } else {
                                None
                            };
                            drop(state);
                            if let Some(peak) = new_peak {
                                self.metrics.update_peak_unique_bytes(peak);
                            }
                            if let Some(start) = wait_start {
                                self.metrics.record_reservation_wait(start.elapsed());
                            }
                            let phase2_res = if phase2_needed == 0 {
                                None
                            } else if phase1_compressed == 0 {
                                compressed_res.take()
                            } else {
                                compressed_res.as_mut().map(|r| r.split(phase2_needed))
                            };
                            let leases = self
                                .split_buffer_leases(coalesced_lengths, compressed_res);
                            let phase2_lease = Phase2BudgetLease {
                                budget: Arc::clone(self),
                                bytes: phase2_needed,
                                reservation: phase2_res,
                            };
                            let decoded_lease = SplitDecodedLease {
                                budget: Arc::clone(self),
                                reservation: decoded_res,
                            };
                            return Ok((leases, phase2_lease, decoded_lease));
                        }
                        Err(err) => {
                            let can_progress = state.in_flight_decode_readers > 0
                                || state.in_flight_filter_planners > 0
                                || state.active_split_lanes
                                    > state.waiting_for_memory + 1;
                            if !can_progress {
                                self.signal_change(&mut state);
                                return Err(err);
                            }
                        }
                    }
                } else {
                    let can_progress = state.in_flight_decode_readers > 0
                        || state.in_flight_filter_planners > 0
                        || state.active_split_lanes > state.waiting_for_memory + 1;
                    if !can_progress {
                        let cap = state.cap;
                        let live = state.live_bytes;
                        self.signal_change(&mut state);
                        return Err(resources_datafusion_err!(
                            "Parquet split buffer cap exhausted (live={live}, needed={compressed_needed}, cap={cap})"
                        ));
                    }
                }
                state.waiting_for_memory += 1;
                wait_guard = WaitGuard {
                    budget: Arc::clone(self),
                };
                state.generation
            };
            wait_start.get_or_insert_with(Instant::now);
            let _ = rx.wait_for(|&g| g != seen).await;
            drop(wait_guard);
        }
    }

    /// Reserve compressed bytes for a Phase-2 fetch on an already-admitted
    /// filtered slice, carving first from its pre-reserved [`Phase2BudgetLease`]
    /// so Phase-2 normally requires zero new reservations or waiting.
    async fn reserve_phase2_fetch(
        self: &Arc<Self>,
        coalesced_lengths: &[usize],
        phase2_lease: Option<&mut Phase2BudgetLease>,
    ) -> Result<Vec<SplitBufferLease>> {
        let compressed_needed: usize = coalesced_lengths.iter().copied().sum();
        if compressed_needed == 0 {
            return Ok(self.split_buffer_leases(coalesced_lengths, None));
        }
        let (covered_bytes, covered_res) = match phase2_lease {
            Some(lease) => {
                if let Some(leases) = lease.take_leases(coalesced_lengths) {
                    return Ok(leases);
                }
                lease.take_all()
            }
            None => (0, None),
        };
        let overflow = compressed_needed.saturating_sub(covered_bytes);
        if overflow == 0 {
            return Ok(self.split_buffer_leases(coalesced_lengths, covered_res));
        }
        // Keep `covered_bytes` under a drop guard so cancelling or erroring
        // while waiting on overflow releases `live_bytes` and pool reservation.
        let mut covered_guard = Phase2BudgetLease {
            budget: Arc::clone(self),
            bytes: covered_bytes,
            reservation: covered_res,
        };
        // Only reached if Phase-2 needed bytes beyond what admission pre-reserved.
        let mut rx = self.changed_rx.clone();
        let mut wait_start: Option<Instant> = None;
        loop {
            let seen = {
                let mut state = self.state.lock();
                let should_throttle_cap = state.live_bytes.saturating_add(overflow)
                    > state.cap
                    && state.in_flight_decode_readers > 0;
                if !should_throttle_cap {
                    match self.try_reserve_pool_single("ParquetSplitBuffer", overflow) {
                        Ok(overflow_res) => {
                            state.live_bytes += overflow;
                            let new_peak = if state.live_bytes > state.peak_live_bytes {
                                state.peak_live_bytes = state.live_bytes;
                                Some(state.peak_live_bytes)
                            } else {
                                None
                            };
                            drop(state);
                            if let Some(peak) = new_peak {
                                self.metrics.update_peak_unique_bytes(peak);
                            }
                            if let Some(start) = wait_start {
                                self.metrics.record_reservation_wait(start.elapsed());
                            }
                            let (_, covered_res) = covered_guard.take_all();
                            let combined = match (covered_res, overflow_res) {
                                (Some(a), Some(b)) => {
                                    let add = b.size();
                                    drop(b);
                                    a.grow(add);
                                    Some(a)
                                }
                                (Some(a), None) => Some(a),
                                (None, Some(b)) => Some(b),
                                (None, None) => None,
                            };
                            return Ok(
                                self.split_buffer_leases(coalesced_lengths, combined)
                            );
                        }
                        Err(err) => {
                            // Other decode readers or peer filter planners can still
                            // complete and release pool capacity.
                            let can_progress = state.in_flight_decode_readers > 0
                                || state.in_flight_filter_planners > 1;
                            if !can_progress {
                                drop(state);
                                drop(covered_guard);
                                return Err(err);
                            }
                        }
                    }
                }
                state.generation
            };
            wait_start.get_or_insert_with(Instant::now);
            let _ = rx.wait_for(|&g| g != seen).await;
        }
    }

    fn try_reserve_decoded_sync(
        self: &Arc<Self>,
        decoded_estimate: usize,
    ) -> Option<SplitDecodedLease> {
        let res = self
            .try_reserve_pool_single("ParquetSplitDecode", decoded_estimate)
            .ok()?;
        Some(SplitDecodedLease {
            budget: Arc::clone(self),
            reservation: res,
        })
    }

    async fn reserve_decoded(
        self: &Arc<Self>,
        decoded_estimate: usize,
    ) -> Result<SplitDecodedLease> {
        let (_, _, decoded_lease) =
            self.reserve_for_fetch(&[], || 0, decoded_estimate).await?;
        Ok(decoded_lease)
    }

    fn try_reserve_pool_single(
        &self,
        name: &str,
        bytes: usize,
    ) -> Result<Option<MemoryReservation>> {
        let Some(pool) = &self.memory_pool else {
            return Ok(None);
        };
        if bytes == 0 {
            return Ok(None);
        }
        let res = MemoryConsumer::new(name).register(pool);
        res.try_grow(bytes)?;
        Ok(Some(res))
    }

    fn try_reserve_pool_pair(
        &self,
        compressed: usize,
        decoded: usize,
    ) -> Result<(Option<MemoryReservation>, Option<MemoryReservation>)> {
        let compressed_res =
            self.try_reserve_pool_single("ParquetSplitBuffer", compressed)?;
        let decoded_res = self.try_reserve_pool_single("ParquetSplitDecode", decoded)?;
        Ok((compressed_res, decoded_res))
    }

    fn split_buffer_leases(
        self: &Arc<Self>,
        lengths: &[usize],
        mut total_res: Option<MemoryReservation>,
    ) -> Vec<SplitBufferLease> {
        let mut leases = Vec::with_capacity(lengths.len());
        for (idx, &len) in lengths.iter().enumerate() {
            let res = if len == 0 {
                None
            } else if idx + 1 == lengths.len() {
                total_res.take()
            } else {
                total_res.as_mut().map(|r| r.split(len))
            };
            leases.push(SplitBufferLease {
                budget: Arc::clone(self),
                bytes: len,
                reservation: res,
            });
        }
        leases
    }

    fn track_decode_reader(self: &Arc<Self>) -> InFlightReaderGuard {
        self.state.lock().in_flight_decode_readers += 1;
        InFlightReaderGuard {
            budget: Arc::clone(self),
            is_filter: false,
        }
    }

    fn track_filter_planner(self: &Arc<Self>) -> InFlightReaderGuard {
        self.state.lock().in_flight_filter_planners += 1;
        InFlightReaderGuard {
            budget: Arc::clone(self),
            is_filter: true,
        }
    }

    #[cfg(test)]
    fn live_bytes(&self) -> usize {
        self.state.lock().live_bytes
    }

    #[cfg(test)]
    fn admitted_frontier_bytes(&self) -> usize {
        self.state.lock().admitted_frontier_bytes
    }
}

struct WaitGuard {
    budget: Arc<SplitBudget>,
}

impl Drop for WaitGuard {
    fn drop(&mut self) {
        let mut state = self.budget.state.lock();
        state.waiting_for_memory = state.waiting_for_memory.saturating_sub(1);
    }
}

struct FrontierGuard {
    budget: Arc<SplitBudget>,
    frontier_bytes: usize,
    pool_frontier_bytes: usize,
}

impl Drop for FrontierGuard {
    fn drop(&mut self) {
        let mut state = self.budget.state.lock();
        state.admitted_frontier_bytes = state
            .admitted_frontier_bytes
            .saturating_sub(self.frontier_bytes);
        state.admitted_pool_frontier_bytes = state
            .admitted_pool_frontier_bytes
            .saturating_sub(self.pool_frontier_bytes);
        state.active_split_lanes = state.active_split_lanes.saturating_sub(1);
        self.budget.signal_change(&mut state);
    }
}

struct SplitBufferLease {
    budget: Arc<SplitBudget>,
    bytes: usize,
    reservation: Option<MemoryReservation>,
}

impl Drop for SplitBufferLease {
    fn drop(&mut self) {
        // Release pool reservation first so woken waiters immediately see
        // reclaimed pool capacity.
        self.reservation = None;
        let mut state = self.budget.state.lock();
        state.live_bytes = state.live_bytes.saturating_sub(self.bytes);
        self.budget.signal_change(&mut state);
    }
}

/// Pre-reserved compressed headroom for a filtered slice's Phase-2 output/predicate
/// fetches, carved into [`SplitBufferLease`]s when Phase-2 runs and automatically
/// releasing any unused bytes when the slice finishes or drops.
struct Phase2BudgetLease {
    budget: Arc<SplitBudget>,
    bytes: usize,
    reservation: Option<MemoryReservation>,
}

impl Phase2BudgetLease {
    fn take_leases(&mut self, lengths: &[usize]) -> Option<Vec<SplitBufferLease>> {
        let needed: usize = lengths.iter().copied().sum();
        if needed > self.bytes {
            return None;
        }
        self.bytes -= needed;
        let mut leases = Vec::with_capacity(lengths.len());
        for &len in lengths {
            let res = if len == 0 {
                None
            } else if self.bytes == 0
                && self.reservation.as_ref().is_some_and(|r| r.size() == len)
            {
                self.reservation.take()
            } else {
                self.reservation.as_mut().map(|r| r.split(len))
            };
            leases.push(SplitBufferLease {
                budget: Arc::clone(&self.budget),
                bytes: len,
                reservation: res,
            });
        }
        Some(leases)
    }

    fn take_all(&mut self) -> (usize, Option<MemoryReservation>) {
        let bytes = std::mem::take(&mut self.bytes);
        let res = self.reservation.take();
        (bytes, res)
    }
}

impl Drop for Phase2BudgetLease {
    fn drop(&mut self) {
        if self.bytes == 0 && self.reservation.is_none() {
            return;
        }
        self.reservation = None;
        let mut state = self.budget.state.lock();
        state.live_bytes = state.live_bytes.saturating_sub(self.bytes);
        self.budget.signal_change(&mut state);
    }
}

struct SplitDecodedLease {
    budget: Arc<SplitBudget>,
    reservation: Option<MemoryReservation>,
}

impl Drop for SplitDecodedLease {
    fn drop(&mut self) {
        self.reservation = None;
        let mut state = self.budget.state.lock();
        self.budget.signal_change(&mut state);
    }
}

struct InFlightReaderGuard {
    budget: Arc<SplitBudget>,
    is_filter: bool,
}

impl Drop for InFlightReaderGuard {
    fn drop(&mut self) {
        let mut state = self.budget.state.lock();
        if self.is_filter {
            state.in_flight_filter_planners =
                state.in_flight_filter_planners.saturating_sub(1);
        } else {
            state.in_flight_decode_readers =
                state.in_flight_decode_readers.saturating_sub(1);
        }
        self.budget.signal_change(&mut state);
    }
}

/// Backing buffer owner wrapped via `Bytes::from_owner` so every zero-copy
/// `Bytes::slice` view across broker, ready queue, and worker decode shares
/// one reservation and releases it only when the final owner drops.
///
/// Field drop order is load-bearing: `data` drops before `_lease` so the
/// backing allocation is freed before `SplitBudget::live_bytes` decrements.
struct TrackedBytes {
    data: Bytes,
    _lease: SplitBufferLease,
}

impl AsRef<[u8]> for TrackedBytes {
    fn as_ref(&self) -> &[u8] {
        self.data.as_ref()
    }
}

fn wrap_tracked_bytes(data: Bytes, lease: SplitBufferLease) -> Bytes {
    Bytes::from_owner(TrackedBytes {
        data,
        _lease: lease,
    })
}

/// View returned from a multi-range `get_byte_ranges` call sharing a single
/// [`SplitBufferLease`] across every non-detached view from that call, so
/// object-store coalesced backing allocations stay accounted while any view
/// can still reference them.
struct SharedTrackedBytes {
    data: Bytes,
    _lease: Arc<SplitBufferLease>,
}

impl AsRef<[u8]> for SharedTrackedBytes {
    fn as_ref(&self) -> &[u8] {
        self.data.as_ref()
    }
}

fn wrap_shared_tracked_bytes(data: Bytes, lease: Arc<SplitBufferLease>) -> Bytes {
    Bytes::from_owner(SharedTrackedBytes {
        data,
        _lease: lease,
    })
}

fn is_flat_fixed_width_leaf(col_desc: &ColumnDescriptor) -> Option<usize> {
    if col_desc.max_rep_level() != 0
        || col_desc.max_def_level() > 1
        || col_desc.path().parts().len() != 1
    {
        return None;
    }
    match col_desc.physical_type() {
        PhysicalType::BOOLEAN => Some(1),
        PhysicalType::INT32 | PhysicalType::FLOAT => Some(4),
        PhysicalType::INT64 | PhysicalType::DOUBLE => Some(8),
        PhysicalType::INT96 => Some(12),
        PhysicalType::FIXED_LEN_BYTE_ARRAY => usize::try_from(col_desc.type_length())
            .ok()
            .filter(|&n| n > 0),
        PhysicalType::BYTE_ARRAY => None,
    }
}

fn validate_offset_index(
    locations: &[PageLocation],
    total_rows: usize,
    chunk_start: u64,
    chunk_len: u64,
) -> bool {
    if locations.is_empty() || total_rows == 0 {
        return false;
    }
    let Some(chunk_end) = chunk_start.checked_add(chunk_len) else {
        return false;
    };
    if locations[0].first_row_index != 0 {
        return false;
    }
    let mut prev_end = chunk_start;
    let mut prev_row: Option<i64> = None;
    for loc in locations {
        if loc.offset < 0 || loc.compressed_page_size <= 0 || loc.first_row_index < 0 {
            return false;
        }
        let Ok(first_row) = usize::try_from(loc.first_row_index) else {
            return false;
        };
        if first_row >= total_rows {
            return false;
        }
        if let Some(prev) = prev_row
            && loc.first_row_index <= prev
        {
            return false;
        }
        let start = loc.offset as u64;
        let Some(end) = start.checked_add(loc.compressed_page_size as u64) else {
            return false;
        };
        if start < prev_end || end > chunk_end {
            return false;
        }
        prev_end = end;
        prev_row = Some(loc.first_row_index);
    }
    true
}

fn dictionary_range(chunk_start: u64, locations: &[PageLocation]) -> Option<Range<u64>> {
    match locations.first() {
        Some(first) if (first.offset as u64) > chunk_start => {
            Some(chunk_start..first.offset as u64)
        }
        _ => None,
    }
}

#[derive(Debug, Clone)]
struct SliceSpec {
    slice_idx: usize,
    row_range: Range<usize>,
    selection: RowSelection,
    selected_rows: usize,
    all_ranges: Vec<(u64, u64)>,
    decoded_estimate: usize,
}

fn push_selector(selectors: &mut Vec<RowSelector>, selector: RowSelector) {
    if selector.row_count == 0 {
        return;
    }
    if let Some(last) = selectors.last_mut()
        && last.skip == selector.skip
    {
        last.row_count += selector.row_count;
    } else {
        selectors.push(selector);
    }
}

fn make_range_selection(lo: usize, hi: usize, total_rows: usize) -> RowSelection {
    let mut selectors = Vec::with_capacity(3);
    push_selector(&mut selectors, RowSelector::skip(lo));
    push_selector(&mut selectors, RowSelector::select(hi - lo));
    push_selector(&mut selectors, RowSelector::skip(total_rows - hi));
    RowSelection::from(selectors)
}

/// Choose disjoint row ranges covering `0..total_rows`, snapping each cut to a
/// page boundary that minimizes compressed bytes of straddled pages across
/// `leaf_locations` (breaking ties by proximity to `lo + target_slice_rows`).
///
/// Searches a tight window `[lo + target/2, lo + 3*target/2]` first so slices
/// stay near `target_slice_rows`; widens to `[lo + target/2, lo + 3*target]`
/// only when no page boundary falls in the tight window (e.g., narrow fixed-width
/// columns with 1 MiB pages holding ~256 Ki rows). Falls back to fixed
/// `target_slice_rows` steps when no page boundary exists in either window.
fn choose_aligned_boundaries(
    total_rows: usize,
    target_slice_rows: usize,
    leaf_locations: &[&[PageLocation]],
) -> Vec<Range<usize>> {
    if total_rows == 0 || target_slice_rows == 0 {
        return Vec::new();
    }
    let min_step = (target_slice_rows / 2).max(1);
    let tight_max_step = target_slice_rows.saturating_add(min_step);
    let wide_max_step = target_slice_rows.saturating_mul(3).max(tight_max_step);

    let mut candidates: Vec<usize> = Vec::new();
    for locs in leaf_locations {
        for loc in *locs {
            if let Ok(row) = usize::try_from(loc.first_row_index)
                && row > 0
                && row < total_rows
            {
                candidates.push(row);
            }
        }
    }
    candidates.sort_unstable();
    candidates.dedup();

    let cut_cost = |cut: usize| -> u64 {
        let mut cost = 0u64;
        for locs in leaf_locations {
            let idx = locs.partition_point(|loc| (loc.first_row_index as usize) < cut);
            if idx < locs.len() && (locs[idx].first_row_index as usize) == cut {
                continue;
            }
            if idx > 0 {
                cost =
                    cost.saturating_add(locs[idx - 1].compressed_page_size.max(0) as u64);
            }
        }
        cost
    };

    let best_in_window = |win_lo: usize, win_hi: usize, ideal: usize| -> Option<usize> {
        if win_lo > win_hi {
            return None;
        }
        let start_idx = candidates.partition_point(|&c| c < win_lo);
        let end_idx = candidates.partition_point(|&c| c <= win_hi);
        candidates[start_idx..end_idx]
            .iter()
            .copied()
            .min_by_key(|&c| (cut_cost(c), c.abs_diff(ideal)))
    };

    let mut ranges = Vec::new();
    let mut lo = 0usize;
    while lo < total_rows {
        let remaining = total_rows - lo;
        if remaining <= target_slice_rows
            || (lo > 0 && remaining < target_slice_rows + min_step)
        {
            ranges.push(lo..total_rows);
            break;
        }
        let ideal = lo.saturating_add(target_slice_rows).min(total_rows);
        let win_lo = lo.saturating_add(min_step);
        let max_cut = total_rows.saturating_sub(min_step);
        let tight_hi = lo.saturating_add(tight_max_step).min(max_cut);
        let wide_hi = lo.saturating_add(wide_max_step).min(max_cut);
        let hi = best_in_window(win_lo, tight_hi, ideal)
            .or_else(|| best_in_window(win_lo, wide_hi, ideal))
            .unwrap_or(ideal);
        let hi = hi.max(lo + 1).min(total_rows);
        ranges.push(lo..hi);
        lo = hi;
    }
    ranges
}

fn split_selection_into_slices_with_ranges(
    existing: Option<&RowSelection>,
    total_rows: usize,
    ranges: &[Range<usize>],
) -> Vec<SliceSpec> {
    if total_rows == 0 || ranges.is_empty() {
        return Vec::new();
    }
    let mut slices = Vec::new();
    match existing {
        None => {
            for r in ranges {
                let selection = make_range_selection(r.start, r.end, total_rows);
                slices.push(SliceSpec {
                    slice_idx: slices.len(),
                    row_range: r.clone(),
                    selection,
                    selected_rows: r.end - r.start,
                    all_ranges: Vec::new(),
                    decoded_estimate: 0,
                });
            }
        }
        Some(existing) => {
            let mut remaining = existing.clone();
            for r in ranges {
                let chunk = remaining.split_off(r.end - r.start);
                if chunk.selects_any() {
                    let selected_rows = chunk.row_count();
                    let mut selectors = Vec::new();
                    push_selector(&mut selectors, RowSelector::skip(r.start));
                    let mut covered = r.start;
                    for sel in chunk.iter() {
                        covered += sel.row_count;
                        push_selector(&mut selectors, *sel);
                    }
                    if covered < total_rows {
                        push_selector(
                            &mut selectors,
                            RowSelector::skip(total_rows - covered),
                        );
                    }
                    slices.push(SliceSpec {
                        slice_idx: slices.len(),
                        row_range: r.clone(),
                        selection: RowSelection::from(selectors),
                        selected_rows,
                        all_ranges: Vec::new(),
                        decoded_estimate: 0,
                    });
                }
            }
        }
    }
    slices
}

#[cfg(test)]
fn split_selection_into_slices(
    existing: Option<&RowSelection>,
    total_rows: usize,
    target_slice_rows: usize,
) -> Vec<SliceSpec> {
    if total_rows == 0 || target_slice_rows == 0 {
        return Vec::new();
    }
    let mut ranges = Vec::new();
    let mut lo = 0;
    while lo < total_rows {
        let hi = (lo + target_slice_rows).min(total_rows);
        ranges.push(lo..hi);
        lo = hi;
    }
    split_selection_into_slices_with_ranges(existing, total_rows, &ranges)
}

struct RowGroupSplitPlan {
    slices: VecDeque<SliceSpec>,
    last_consumer: HashMap<(u64, u64), usize>,
    shared_ranges_per_slice: Vec<Vec<(u64, u64)>>,
    shared_consumer_counts: HashMap<(u64, u64), usize>,
    frontier_bytes: usize,
    pool_frontier_bytes: usize,
}

#[expect(clippy::too_many_arguments)]
fn plan_row_group_splits(
    metadata: &ParquetMetaData,
    rg_index: usize,
    existing_selection: Option<&RowSelection>,
    reads_leaf: impl Fn(usize) -> bool,
    is_cached_leaf: impl Fn(usize) -> bool,
    config: &ParquetSplitConfig,
    enable_page_index: bool,
    batch_size: usize,
    pool_limit: Option<usize>,
) -> std::result::Result<RowGroupSplitPlan, UnsplitReason> {
    if !config.enabled || config.target_slice_rows == 0 {
        return Err(UnsplitReason::Disabled);
    }
    if !enable_page_index {
        return Err(UnsplitReason::PageIndexDisabled);
    }
    let rg_meta = metadata.row_group(rg_index);
    let Ok(total_rows) = usize::try_from(rg_meta.num_rows()) else {
        return Err(UnsplitReason::BelowThreshold);
    };
    if total_rows < config.min_row_group_rows || total_rows <= config.target_slice_rows {
        return Err(UnsplitReason::BelowThreshold);
    }

    let schema_descr = metadata.file_metadata().schema_descr();
    let mut read_leaves: Vec<(usize, usize)> = Vec::new();
    for leaf_idx in 0..schema_descr.num_columns() {
        if !reads_leaf(leaf_idx) {
            continue;
        }
        let Some(byte_width) = is_flat_fixed_width_leaf(&schema_descr.column(leaf_idx))
        else {
            return Err(UnsplitReason::UnsupportedColumnType);
        };
        read_leaves.push((leaf_idx, byte_width));
    }
    if read_leaves.is_empty() {
        return Err(UnsplitReason::MetadataOnly);
    }

    let Some(page_index) = metadata.page_index() else {
        return Err(UnsplitReason::MissingOffsetIndex);
    };
    struct LeafOffsetInfo<'a> {
        locations: &'a [PageLocation],
        dict_range: Option<Range<u64>>,
        cached: bool,
        byte_width: usize,
    }
    let mut leaf_offsets: Vec<LeafOffsetInfo<'_>> = Vec::with_capacity(read_leaves.len());
    let mut any_cached_leaf = false;
    let mut total_one_page_scratch: usize = 0;
    for &(leaf_idx, byte_width) in &read_leaves {
        let Some(offset_index) = page_index.offset_index(rg_index, leaf_idx) else {
            return Err(UnsplitReason::MissingOffsetIndex);
        };
        let locations = offset_index.page_locations().as_slice();
        let (chunk_start, chunk_len) = rg_meta.column(leaf_idx).byte_range();
        if !validate_offset_index(locations, total_rows, chunk_start, chunk_len) {
            return Err(UnsplitReason::InvalidOffsetIndex);
        }
        let cached = is_cached_leaf(leaf_idx);
        any_cached_leaf |= cached;
        // ParquetRecordBatchReader decodes one decompressed page per leaf at a
        // time, so size per-slice scratch to one average page per leaf rather
        // than the whole row group's uncompressed size.
        let chunk_uncompressed =
            rg_meta.column(leaf_idx).uncompressed_size().max(0) as u64;
        let num_pages = locations.len().max(1) as u64;
        let one_page_uncompressed = chunk_uncompressed
            .div_ceil(num_pages)
            .min(usize::MAX as u64) as usize;
        total_one_page_scratch =
            total_one_page_scratch.saturating_add(one_page_uncompressed);
        leaf_offsets.push(LeafOffsetInfo {
            locations,
            dict_range: dictionary_range(chunk_start, locations),
            cached,
            byte_width,
        });
    }

    let leaf_locations: Vec<&[PageLocation]> =
        leaf_offsets.iter().map(|info| info.locations).collect();
    let row_ranges =
        choose_aligned_boundaries(total_rows, config.target_slice_rows, &leaf_locations);
    let mut slices = split_selection_into_slices_with_ranges(
        existing_selection,
        total_rows,
        &row_ranges,
    );
    if slices.len() < 2 {
        return Err(UnsplitReason::SingleSlice);
    }

    // Collect exact per-slice byte ranges (plus batch-boundary expansion only
    // for leaves that RowGroupCache may widen during predicate evaluation) and
    // per-slice decoded workspace estimates.
    let mut per_slice_ranges: Vec<Vec<(u64, u64)>> = Vec::with_capacity(slices.len());
    let mut last_consumer: HashMap<(u64, u64), usize> = HashMap::new();
    let mut consumer_counts: HashMap<(u64, u64), usize> = HashMap::new();
    let mut sum_slice_bytes: usize = 0;

    for slice in &mut slices {
        let slice_batch_rows = batch_size.min(slice.selected_rows.max(1));
        let batch_workspace: usize = leaf_offsets
            .iter()
            .map(|info| {
                slice_batch_rows.saturating_mul(info.byte_width.saturating_add(3))
            })
            .fold(0usize, usize::saturating_add);
        let decoded_estimate = batch_workspace.saturating_add(total_one_page_scratch);
        if decoded_estimate > config.max_decoded_bytes {
            return Err(UnsplitReason::DecodedWorkspaceCap);
        }
        slice.decoded_estimate = decoded_estimate;

        let expanded_sel = if any_cached_leaf && batch_size > 0 {
            let exp_lo = (slice.row_range.start / batch_size) * batch_size;
            let exp_hi = slice
                .row_range
                .end
                .div_ceil(batch_size)
                .saturating_mul(batch_size)
                .min(total_rows);
            Some(make_range_selection(exp_lo, exp_hi, total_rows))
        } else {
            None
        };
        let mut ranges = Vec::new();
        let mut seen_in_slice = HashSet::new();
        let mut slice_bytes: usize = 0;
        for info in &leaf_offsets {
            let sel = if info.cached {
                expanded_sel.as_ref().unwrap_or(&slice.selection)
            } else {
                &slice.selection
            };
            let page_ranges = sel.scan_ranges(info.locations);
            if page_ranges.is_empty() {
                continue;
            }
            if let Some(dict) = &info.dict_range {
                let key = (dict.start, dict.end);
                if seen_in_slice.insert(key) {
                    slice_bytes =
                        slice_bytes.saturating_add((dict.end - dict.start) as usize);
                    ranges.push(key);
                }
            }
            for r in page_ranges {
                let key = (r.start, r.end);
                if seen_in_slice.insert(key) {
                    slice_bytes = slice_bytes.saturating_add((r.end - r.start) as usize);
                    ranges.push(key);
                }
            }
        }
        if slice_bytes > config.max_slice_compressed_bytes {
            return Err(UnsplitReason::SliceCompressedCap);
        }
        sum_slice_bytes = sum_slice_bytes.saturating_add(slice_bytes);
        for &key in &ranges {
            last_consumer.insert(key, slice.slice_idx);
            *consumer_counts.entry(key).or_insert(0) += 1;
        }
        slice.all_ranges.clone_from(&ranges);
        per_slice_ranges.push(ranges);
    }

    let unique_bytes: usize = last_consumer
        .keys()
        .map(|&(start, end)| (end - start) as usize)
        .sum();
    if unique_bytes > 0
        && (sum_slice_bytes as f64)
            > config.max_decode_amplification * (unique_bytes as f64)
    {
        return Err(UnsplitReason::DecodeAmplification);
    }

    // Simulate sequential slice progression to compute the exact peak broker
    // frontier (retained shared pages/dicts + incremental miss bytes for one
    // slice + temporary detached copies of newly retained ranges when the same
    // fetch also contains ephemeral ranges) and peak memory-pool footprint.
    let mut cached: HashMap<(u64, u64), usize> = HashMap::new();
    let mut cached_bytes: usize = 0;
    let mut frontier_bytes: usize = 0;
    let mut pool_frontier_bytes: usize = 0;
    for (slice_idx, ranges) in per_slice_ranges.iter().enumerate() {
        let mut new_bytes: usize = 0;
        let mut newly_retained_bytes: usize = 0;
        let mut has_ephemeral_miss = false;
        for &key in ranges {
            if !cached.contains_key(&key) {
                let len = (key.1 - key.0) as usize;
                new_bytes = new_bytes.saturating_add(len);
                if last_consumer[&key] > slice_idx {
                    newly_retained_bytes = newly_retained_bytes.saturating_add(len);
                } else {
                    has_ephemeral_miss = true;
                }
            }
        }
        let extra_detach_bytes = if has_ephemeral_miss {
            newly_retained_bytes
        } else {
            0
        };
        let compressed_at_slice = cached_bytes
            .saturating_add(new_bytes)
            .saturating_add(extra_detach_bytes);
        frontier_bytes = frontier_bytes.max(compressed_at_slice);
        pool_frontier_bytes = pool_frontier_bytes
            .max(compressed_at_slice.saturating_add(slices[slice_idx].decoded_estimate));
        for &key in ranges {
            let last = last_consumer[&key];
            if last > slice_idx && !cached.contains_key(&key) {
                let len = (key.1 - key.0) as usize;
                cached.insert(key, last);
                cached_bytes += len;
            }
        }
        cached.retain(|&key, &mut last| {
            if last <= slice_idx {
                cached_bytes -= (key.1 - key.0) as usize;
                false
            } else {
                true
            }
        });
    }
    if frontier_bytes > config.split_buffer_cap_bytes {
        return Err(UnsplitReason::BufferCap);
    }
    if pool_limit.is_some_and(|limit| pool_frontier_bytes > limit) {
        return Err(UnsplitReason::MemoryPoolCap);
    }

    let shared_consumer_counts: HashMap<(u64, u64), usize> = consumer_counts
        .into_iter()
        .filter(|&(_, count)| count > 1)
        .collect();
    let shared_ranges_per_slice: Vec<Vec<(u64, u64)>> = per_slice_ranges
        .into_iter()
        .map(|ranges| {
            ranges
                .into_iter()
                .filter(|key| shared_consumer_counts.contains_key(key))
                .collect()
        })
        .collect();

    Ok(RowGroupSplitPlan {
        slices: slices.into(),
        last_consumer,
        shared_ranges_per_slice,
        shared_consumer_counts,
        frontier_bytes,
        pool_frontier_bytes,
    })
}

struct CoalescedRequest {
    range: Range<u64>,
    members: Vec<Range<u64>>,
}

/// Merge strictly adjacent missed ranges into bounded requests without fetching
/// unused gaps. Ranges retained for later slices (`last_consumer > slice_idx`)
/// are only merged with ranges that share the same `last_consumer`, so a
/// short-lived interior page is never pinned by a dictionary or boundary page.
fn coalesce_misses(
    mut misses: Vec<Range<u64>>,
    last_consumer: &HashMap<(u64, u64), usize>,
    slice_idx: usize,
    max_coalesced_bytes: usize,
) -> Vec<CoalescedRequest> {
    if misses.is_empty() {
        return Vec::new();
    }
    misses.sort_unstable_by_key(|r| (r.start, r.end));
    misses.dedup();

    let mut groups: Vec<CoalescedRequest> = Vec::new();
    let mut group_last: usize = 0;

    for r in misses {
        let r_last = last_consumer
            .get(&(r.start, r.end))
            .copied()
            .unwrap_or(slice_idx);
        let r_retained = r_last > slice_idx;
        let can_merge = if let Some(last_group) = groups.last() {
            let group_retained = group_last > slice_idx;
            let lifetime_compatible = if group_retained || r_retained {
                group_last == r_last
            } else {
                true
            };
            last_group.range.end == r.start
                && lifetime_compatible
                && ((r.end - last_group.range.start) as usize) <= max_coalesced_bytes
        } else {
            false
        };
        if can_merge {
            let last_group = groups.last_mut().unwrap();
            last_group.range.end = r.end;
            last_group.members.push(r);
        } else {
            group_last = r_last;
            groups.push(CoalescedRequest {
                range: r.clone(),
                members: vec![r],
            });
        }
    }
    groups
}

/// Coalesce strictly adjacent non-shared ranges (for Phase-2 interior fetches).
fn coalesce_adjacent_ranges(
    mut misses: Vec<Range<u64>>,
    max_coalesced_bytes: usize,
) -> Vec<CoalescedRequest> {
    if misses.is_empty() {
        return Vec::new();
    }
    misses.sort_unstable_by_key(|r| (r.start, r.end));
    misses.dedup();
    let mut groups: Vec<CoalescedRequest> = Vec::new();
    for r in misses {
        let can_merge = groups.last().is_some_and(|g| {
            g.range.end == r.start
                && ((r.end - g.range.start) as usize) <= max_coalesced_bytes
        });
        if can_merge {
            let g = groups.last_mut().unwrap();
            g.range.end = r.end;
            g.members.push(r);
        } else {
            groups.push(CoalescedRequest {
                range: r.clone(),
                members: vec![r],
            });
        }
    }
    groups
}

/// Build reservation lengths for one `RangeBroker` `get_byte_ranges` call:
/// - If any member is ephemeral (`last_consumer <= slice_idx`), lease 0 covers
///   the entire call's bytes and is shared across all ephemeral views from that
///   call (object stores may merge separately requested ranges into one buffer).
/// - Each retained member (`last_consumer > slice_idx`) gets its own lease so
///   it is copied into a standalone buffer before caching and never pins the
///   object store's coalesced backing allocation past the current slice.
fn plan_range_broker_leases(
    coalesced: &[CoalescedRequest],
    last_consumer: &HashMap<(u64, u64), usize>,
    slice_idx: usize,
) -> (Vec<usize>, bool) {
    let call_total: usize = coalesced
        .iter()
        .map(|c| (c.range.end - c.range.start) as usize)
        .sum();
    let mut retained_lengths = Vec::new();
    let mut has_ephemeral = false;
    for req in coalesced {
        for member in &req.members {
            let last = last_consumer
                .get(&(member.start, member.end))
                .copied()
                .unwrap_or(slice_idx);
            if last > slice_idx {
                retained_lengths.push((member.end - member.start) as usize);
            } else {
                has_ephemeral = true;
            }
        }
    }
    let mut lease_lengths =
        Vec::with_capacity(usize::from(has_ephemeral) + retained_lengths.len());
    if has_ephemeral {
        lease_lengths.push(call_total);
    }
    lease_lengths.extend(retained_lengths);
    (lease_lengths, has_ephemeral)
}

/// Row-group-local range broker owned solely by the file I/O lane (no lock).
struct RangeBroker {
    entries: HashMap<(u64, u64), (Bytes, usize)>,
}

impl RangeBroker {
    fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Split `requested` into already-cached views and sorted missed ranges.
    fn partition_misses(&self, requested: &[Range<u64>]) -> Vec<Range<u64>> {
        requested
            .iter()
            .filter(|r| !self.entries.contains_key(&(r.start, r.end)))
            .cloned()
            .collect()
    }

    /// Ingest newly fetched coalesced buffers, detach and cache entries still
    /// needed after `slice_idx` under individual leases, attach one shared call
    /// lease to all ephemeral views from this `get_byte_ranges` call, assemble
    /// the exact `Vec<Bytes>` matching `requested`, and evict entries whose
    /// `last_consumer <= slice_idx`.
    #[expect(clippy::too_many_arguments)]
    fn complete_fetch(
        &mut self,
        requested: &[Range<u64>],
        coalesced: &[CoalescedRequest],
        raw_bytes: Vec<Bytes>,
        leases: Vec<SplitBufferLease>,
        has_ephemeral: bool,
        last_consumer: &HashMap<(u64, u64), usize>,
        slice_idx: usize,
        metrics: &SplitMetrics,
        err_ctx: &str,
    ) -> Result<Vec<Bytes>> {
        if raw_bytes.len() != coalesced.len() {
            return Err(internal_datafusion_err!(
                "Storage returned {} ranges for {} coalesced requests ({err_ctx})",
                raw_bytes.len(),
                coalesced.len()
            ));
        }
        let mut lease_iter = leases.into_iter();
        let call_lease = has_ephemeral.then(|| {
            Arc::new(
                lease_iter
                    .next()
                    .expect("missing call lease for ephemeral views"),
            )
        });
        let mut fetched_map: HashMap<(u64, u64), Bytes> = HashMap::new();
        let mut total_returned = 0usize;
        for (req, buf) in coalesced.iter().zip(raw_bytes) {
            let expected_len = (req.range.end - req.range.start) as usize;
            if buf.len() != expected_len {
                return Err(internal_datafusion_err!(
                    "Storage returned {} bytes for range {}..{} (expected {expected_len}) ({err_ctx})",
                    buf.len(),
                    req.range.start,
                    req.range.end
                ));
            }
            total_returned += buf.len();
            let req_has_ephemeral = req.members.iter().any(|m| {
                last_consumer
                    .get(&(m.start, m.end))
                    .copied()
                    .unwrap_or(slice_idx)
                    <= slice_idx
            });
            let shared_buf = req_has_ephemeral.then(|| {
                wrap_shared_tracked_bytes(
                    buf.clone(),
                    Arc::clone(call_lease.as_ref().expect("call lease present")),
                )
            });
            for member in &req.members {
                let rel_start = (member.start - req.range.start) as usize;
                let rel_end = (member.end - req.range.start) as usize;
                let key = (member.start, member.end);
                let last = last_consumer.get(&key).copied().unwrap_or(slice_idx);
                if last > slice_idx {
                    // Detach cached bytes into an independent buffer so this
                    // cache entry never pins the store's coalesced allocation.
                    let detached = Bytes::copy_from_slice(&buf[rel_start..rel_end]);
                    let member_lease = lease_iter
                        .next()
                        .expect("missing individual lease for retained range");
                    let tracked = wrap_tracked_bytes(detached, member_lease);
                    self.entries.insert(key, (tracked.clone(), last));
                    fetched_map.insert(key, tracked);
                } else {
                    let view = shared_buf
                        .as_ref()
                        .expect("shared_buf created for ephemeral member")
                        .slice(rel_start..rel_end);
                    fetched_map.insert(key, view);
                }
            }
        }
        drop(call_lease);
        if !coalesced.is_empty() {
            metrics.record_storage_fetch(coalesced.len(), total_returned);
        }

        let mut output = Vec::with_capacity(requested.len());
        let mut reused = 0usize;
        for r in requested {
            let key = (r.start, r.end);
            if let Some(view) = fetched_map.get(&key) {
                output.push(view.clone());
            } else if let Some((view, _)) = self.entries.get(&key) {
                reused += view.len();
                output.push(view.clone());
            } else {
                return Err(internal_datafusion_err!(
                    "Missing resolved byte range {}..{} ({err_ctx})",
                    r.start,
                    r.end
                ));
            }
        }
        metrics.record_reused_bytes(reused);
        self.evict_through(slice_idx);
        Ok(output)
    }

    fn serve_hits_only(
        &mut self,
        requested: &[Range<u64>],
        slice_idx: usize,
        metrics: &SplitMetrics,
        err_ctx: &str,
    ) -> Result<Vec<Bytes>> {
        let mut output = Vec::with_capacity(requested.len());
        let mut reused = 0usize;
        for r in requested {
            let key = (r.start, r.end);
            let Some((view, _)) = self.entries.get(&key) else {
                return Err(internal_datafusion_err!(
                    "Expected cached byte range {}..{} ({err_ctx})",
                    r.start,
                    r.end
                ));
            };
            reused += view.len();
            output.push(view.clone());
        }
        metrics.record_reused_bytes(reused);
        self.evict_through(slice_idx);
        Ok(output)
    }

    fn evict_through(&mut self, slice_idx: usize) {
        self.entries.retain(|_, (_, last)| *last > slice_idx);
    }
}

/// Shared RG-local async range broker for `FilteredBufferedReaders` so Phase-1
/// and concurrent Phase-2 lanes deduplicate shared boundary/dictionary pages
/// without holding a lock across storage I/O.
struct AsyncRangeBroker {
    state: Mutex<HashMap<(u64, u64), AsyncSharedEntry>>,
    split_context: Arc<SplitContext>,
}

struct AsyncSharedEntry {
    remaining_consumers: usize,
    status: AsyncRangeStatus,
}

enum AsyncRangeStatus {
    Unfetched,
    InFlight(watch::Receiver<bool>),
    Ready(Bytes),
    Failed(String),
}

enum RangeAdmission<'a> {
    /// Phase-1 admission on the file lane: reserve Phase-1 misses, uncached
    /// Phase-2 candidate headroom, and decoded workspace atomically.
    Phase1 {
        phase2_candidates: &'a [(u64, u64)],
        decoded_estimate: usize,
        out_leases: &'a mut Option<(Phase2BudgetLease, SplitDecodedLease)>,
    },
    /// Phase-2 fetch on an already-admitted filtered slice: carve from its
    /// pre-reserved [`Phase2BudgetLease`].
    Phase2 {
        phase2_lease: Option<&'a mut Phase2BudgetLease>,
    },
    /// Follow-up file-lane fetch where the decoded lease is already held.
    Phase1FollowUp,
}

impl AsyncRangeBroker {
    fn new(
        shared_consumer_counts: HashMap<(u64, u64), usize>,
        split_context: Arc<SplitContext>,
    ) -> Self {
        let map = shared_consumer_counts
            .into_iter()
            .map(|(key, count)| {
                (
                    key,
                    AsyncSharedEntry {
                        remaining_consumers: count,
                        status: AsyncRangeStatus::Unfetched,
                    },
                )
            })
            .collect();
        Self {
            state: Mutex::new(map),
            split_context,
        }
    }

    /// Total bytes among `candidates` that are not yet `Ready` in this broker,
    /// including temporary detached-copy headroom when a Phase-2 call contains
    /// both private and uncached shared ranges.
    fn uncached_bytes(&self, candidates: &[(u64, u64)]) -> usize {
        if candidates.is_empty() {
            return 0;
        }
        let map = self.state.lock();
        let mut private_bytes = 0usize;
        let mut shared_bytes = 0usize;
        for key in candidates {
            let len = (key.1 - key.0) as usize;
            match map.get(key) {
                Some(entry) if matches!(entry.status, AsyncRangeStatus::Ready(_)) => {}
                Some(_) => {
                    shared_bytes = shared_bytes.saturating_add(len);
                }
                None => {
                    private_bytes = private_bytes.saturating_add(len);
                }
            }
        }
        let extra_detach = if private_bytes > 0 { shared_bytes } else { 0 };
        private_bytes
            .saturating_add(shared_bytes)
            .saturating_add(extra_detach)
    }

    /// Release unconsumed shared range claims when a slice finishes (e.g.,
    /// filtered to 0 rows) or is dropped on cancellation/error.
    fn release_unconsumed(&self, keys: &mut HashSet<(u64, u64)>) {
        if keys.is_empty() {
            return;
        }
        // Drop removed entries (which may hold TrackedBytes leases) outside
        // `self.state` lock so SplitBufferLease::drop never nests locks.
        let mut evicted = Vec::new();
        {
            let mut map = self.state.lock();
            for key in keys.drain() {
                if let Some(entry) = map.get_mut(&key) {
                    entry.remaining_consumers =
                        entry.remaining_consumers.saturating_sub(1);
                    if entry.remaining_consumers == 0
                        && let Some(removed) = map.remove(&key)
                    {
                        evicted.push(removed);
                    }
                }
            }
        }
        drop(evicted);
    }

    /// Resolve `requested` ranges for one slice using `fetch_fn` for misses.
    /// Shared ranges are fetched at most once and deduplicated across in-flight
    /// callers; non-shared ranges are coalesced directly without map overhead.
    async fn resolve_ranges<F, Fut>(
        self: &Arc<Self>,
        requested: &[Range<u64>],
        tracker: &mut Phase2SliceTracker,
        mut admission: RangeAdmission<'_>,
        err_ctx: &str,
        fetch_fn: F,
    ) -> Result<Vec<Bytes>>
    where
        F: Fn(Vec<Range<u64>>) -> Fut,
        Fut: Future<Output = Result<Vec<Bytes>>>,
    {
        let mut resolved: HashMap<(u64, u64), Bytes> = HashMap::new();
        let mut to_wait: Vec<(u64, u64)> = Vec::new();
        let mut shared_to_fetch: Vec<((u64, u64), watch::Sender<bool>)> = Vec::new();
        let mut private_to_fetch: Vec<Range<u64>> = Vec::new();
        let mut reused_bytes = 0usize;

        let mut evicted_entries = Vec::new();
        {
            let mut map = self.state.lock();
            for r in requested {
                let key = (r.start, r.end);
                if resolved.contains_key(&key) {
                    continue;
                }
                if let Some(entry) = map.get_mut(&key) {
                    match &entry.status {
                        AsyncRangeStatus::Ready(bytes) => {
                            let view = bytes.clone();
                            reused_bytes += view.len();
                            if tracker.consume_key(key) {
                                entry.remaining_consumers =
                                    entry.remaining_consumers.saturating_sub(1);
                                if entry.remaining_consumers == 0
                                    && let Some(removed) = map.remove(&key)
                                {
                                    evicted_entries.push(removed);
                                }
                            }
                            resolved.insert(key, view);
                        }
                        AsyncRangeStatus::InFlight(_) => {
                            to_wait.push(key);
                        }
                        AsyncRangeStatus::Unfetched => {
                            let (tx, rx) = watch::channel(false);
                            entry.status = AsyncRangeStatus::InFlight(rx);
                            shared_to_fetch.push((key, tx));
                        }
                        AsyncRangeStatus::Failed(msg) => {
                            return Err(DataFusionError::Execution(format!(
                                "{msg} ({err_ctx})"
                            )));
                        }
                    }
                } else {
                    private_to_fetch.push(r.clone());
                }
            }
        }
        drop(evicted_entries);
        self.split_context.metrics.record_reused_bytes(reused_bytes);

        // Guard ensures any shared ranges we claimed as InFlight revert to
        // Unfetched and wake waiters if this future is dropped/cancelled mid-I/O.
        struct SharedFetchGuard {
            broker: Arc<AsyncRangeBroker>,
            pending: Vec<((u64, u64), watch::Sender<bool>)>,
        }
        impl Drop for SharedFetchGuard {
            fn drop(&mut self) {
                if self.pending.is_empty() {
                    return;
                }
                let mut map = self.broker.state.lock();
                for (key, tx) in self.pending.drain(..) {
                    if let Some(entry) = map.get_mut(&key)
                        && matches!(entry.status, AsyncRangeStatus::InFlight(_))
                    {
                        entry.status = AsyncRangeStatus::Unfetched;
                    }
                    let _ = tx.send(true);
                }
            }
        }
        let shared_keys_only: Vec<(u64, u64)> =
            shared_to_fetch.iter().map(|&(k, _)| k).collect();
        let mut guard = SharedFetchGuard {
            broker: Arc::clone(self),
            pending: shared_to_fetch,
        };

        // Coalesce private misses (bounded by max_coalesced_request_bytes); keep
        // shared ranges as individual requests so their TrackedBytes lifetimes
        // stay independent of short-lived interior pages.
        let mut coalesced = coalesce_adjacent_ranges(
            private_to_fetch,
            self.split_context.config.max_coalesced_request_bytes,
        );
        let private_count = coalesced.len();
        for &(start, end) in &shared_keys_only {
            let range = start..end;
            coalesced.push(CoalescedRequest {
                range: range.clone(),
                members: vec![range],
            });
        }

        let call_total_bytes: usize = coalesced
            .iter()
            .map(|c| (c.range.end - c.range.start) as usize)
            .sum();
        let has_private = private_count > 0;
        let shared_lengths: Vec<usize> = coalesced[private_count..]
            .iter()
            .map(|c| (c.range.end - c.range.start) as usize)
            .collect();
        let mut lease_lengths =
            Vec::with_capacity(usize::from(has_private) + shared_lengths.len());
        if has_private {
            lease_lengths.push(call_total_bytes);
        }
        lease_lengths.extend(shared_lengths);

        let leases = match &mut admission {
            RangeAdmission::Phase1 {
                phase2_candidates,
                decoded_estimate,
                out_leases,
            } => {
                let (leases, phase2_lease, decoded_lease) = self
                    .split_context
                    .budget
                    .reserve_for_fetch(
                        &lease_lengths,
                        || self.uncached_bytes(phase2_candidates),
                        *decoded_estimate,
                    )
                    .await?;
                **out_leases = Some((phase2_lease, decoded_lease));
                leases
            }
            RangeAdmission::Phase2 { phase2_lease } => {
                if coalesced.is_empty() {
                    Vec::new()
                } else {
                    self.split_context
                        .budget
                        .reserve_phase2_fetch(&lease_lengths, phase2_lease.as_deref_mut())
                        .await?
                }
            }
            RangeAdmission::Phase1FollowUp => {
                if coalesced.is_empty() {
                    Vec::new()
                } else {
                    let (leases, _, _) = self
                        .split_context
                        .budget
                        .reserve_for_fetch(&lease_lengths, || 0, 0)
                        .await?;
                    leases
                }
            }
        };

        if !coalesced.is_empty() {
            let fetch_ranges: Vec<Range<u64>> =
                coalesced.iter().map(|c| c.range.clone()).collect();
            let raw_bytes = match fetch_fn(fetch_ranges).await {
                Ok(bytes) => bytes,
                Err(err) => {
                    let msg = err.to_string();
                    let mut map = self.state.lock();
                    for (key, tx) in guard.pending.drain(..) {
                        if let Some(entry) = map.get_mut(&key) {
                            entry.status = AsyncRangeStatus::Failed(msg.clone());
                        }
                        let _ = tx.send(true);
                    }
                    return Err(err.context(err_ctx.to_string()));
                }
            };
            if raw_bytes.len() != coalesced.len() {
                return Err(internal_datafusion_err!(
                    "Storage returned {} ranges for {} coalesced requests ({err_ctx})",
                    raw_bytes.len(),
                    coalesced.len()
                ));
            }
            let mut lease_iter = leases.into_iter();
            let call_lease = has_private.then(|| {
                Arc::new(
                    lease_iter
                        .next()
                        .expect("missing call lease for private views"),
                )
            });
            let mut total_returned = 0usize;
            let mut completed_shared: HashMap<(u64, u64), Bytes> =
                HashMap::with_capacity(shared_keys_only.len());
            for (idx, (req, buf)) in coalesced.iter().zip(raw_bytes).enumerate() {
                let expected_len = (req.range.end - req.range.start) as usize;
                if buf.len() != expected_len {
                    return Err(internal_datafusion_err!(
                        "Storage returned {} bytes for range {}..{} (expected {expected_len}) ({err_ctx})",
                        buf.len(),
                        req.range.start,
                        req.range.end
                    ));
                }
                total_returned += buf.len();
                if idx < private_count {
                    let tracked = wrap_shared_tracked_bytes(
                        buf,
                        Arc::clone(call_lease.as_ref().expect("call lease present")),
                    );
                    for member in &req.members {
                        let rel_start = (member.start - req.range.start) as usize;
                        let rel_end = (member.end - req.range.start) as usize;
                        resolved.insert(
                            (member.start, member.end),
                            tracked.slice(rel_start..rel_end),
                        );
                    }
                } else {
                    // Detach shared bytes into an independent buffer so `Ready`
                    // cache entries never pin the store's coalesced allocation.
                    let detached = Bytes::copy_from_slice(&buf);
                    let shared_lease = lease_iter
                        .next()
                        .expect("missing individual lease for shared range");
                    let tracked = wrap_tracked_bytes(detached, shared_lease);
                    let key = (req.range.start, req.range.end);
                    resolved.insert(key, tracked.clone());
                    completed_shared.insert(key, tracked);
                }
            }
            drop(call_lease);
            self.split_context
                .metrics
                .record_storage_fetch(coalesced.len(), total_returned);

            let mut evicted = Vec::new();
            {
                let mut map = self.state.lock();
                for (key, tx) in guard.pending.drain(..) {
                    if let Some(tracked) = completed_shared.remove(&key)
                        && let Some(entry) = map.get_mut(&key)
                    {
                        entry.status = AsyncRangeStatus::Ready(tracked);
                        if tracker.consume_key(key) {
                            entry.remaining_consumers =
                                entry.remaining_consumers.saturating_sub(1);
                            if entry.remaining_consumers == 0
                                && let Some(removed) = map.remove(&key)
                            {
                                evicted.push(removed);
                            }
                        }
                    }
                    let _ = tx.send(true);
                }
            }
            drop(evicted);
        }

        // Wait for any shared ranges that were already in flight on another lane.
        enum WaitAction {
            Ready(Bytes),
            Wait(watch::Receiver<bool>),
            Takeover(watch::Sender<bool>),
        }
        let mut waited_reused = 0usize;
        for key in to_wait {
            loop {
                let mut evicted = None;
                let action = {
                    let mut map = self.state.lock();
                    match map.get_mut(&key) {
                        Some(entry) => match &entry.status {
                            AsyncRangeStatus::Ready(bytes) => {
                                let view = bytes.clone();
                                if tracker.consume_key(key) {
                                    entry.remaining_consumers =
                                        entry.remaining_consumers.saturating_sub(1);
                                    if entry.remaining_consumers == 0 {
                                        evicted = map.remove(&key);
                                    }
                                }
                                WaitAction::Ready(view)
                            }
                            AsyncRangeStatus::InFlight(rx) => {
                                WaitAction::Wait(rx.clone())
                            }
                            AsyncRangeStatus::Failed(msg) => {
                                return Err(DataFusionError::Execution(format!(
                                    "{msg} ({err_ctx})"
                                )));
                            }
                            AsyncRangeStatus::Unfetched => {
                                // Previous fetcher was cancelled before completing;
                                // take over fetching this range.
                                let (tx, rx) = watch::channel(false);
                                entry.status = AsyncRangeStatus::InFlight(rx);
                                WaitAction::Takeover(tx)
                            }
                        },
                        None => {
                            return Err(internal_datafusion_err!(
                                "Shared range {}..{} evicted while in flight ({err_ctx})",
                                key.0,
                                key.1
                            ));
                        }
                    }
                };
                drop(evicted);
                match action {
                    WaitAction::Ready(view) => {
                        waited_reused += view.len();
                        resolved.insert(key, view);
                        break;
                    }
                    WaitAction::Wait(mut rx) => {
                        let _ = rx.wait_for(|&done| done).await;
                    }
                    WaitAction::Takeover(tx) => {
                        let mut takeover_guard = SharedFetchGuard {
                            broker: Arc::clone(self),
                            pending: vec![(key, tx)],
                        };
                        let len = (key.1 - key.0) as usize;
                        let mut leases = match &mut admission {
                            RangeAdmission::Phase2 { phase2_lease } => {
                                self.split_context
                                    .budget
                                    .reserve_phase2_fetch(
                                        &[len],
                                        phase2_lease.as_deref_mut(),
                                    )
                                    .await?
                            }
                            RangeAdmission::Phase1 { .. }
                            | RangeAdmission::Phase1FollowUp => {
                                let (l, _, _) = self
                                    .split_context
                                    .budget
                                    .reserve_for_fetch(&[len], || 0, 0)
                                    .await?;
                                l
                            }
                        };
                        let lease = leases.pop().unwrap();
                        let range = key.0..key.1;
                        let mut raw = fetch_fn(vec![range]).await?;
                        let buf = raw.pop().ok_or_else(|| {
                            internal_datafusion_err!(
                                "Empty storage response for range {}..{} ({err_ctx})",
                                key.0,
                                key.1
                            )
                        })?;
                        if buf.len() != len {
                            return Err(internal_datafusion_err!(
                                "Storage returned {} bytes for range {}..{} (expected {len}) ({err_ctx})",
                                buf.len(),
                                key.0,
                                key.1
                            ));
                        }
                        self.split_context.metrics.record_storage_fetch(1, len);
                        let detached = Bytes::copy_from_slice(&buf);
                        drop(buf);
                        let tracked = wrap_tracked_bytes(detached, lease);
                        resolved.insert(key, tracked.clone());
                        let mut evicted = Vec::new();
                        {
                            let mut map = self.state.lock();
                            for (k, tx) in takeover_guard.pending.drain(..) {
                                if let Some(entry) = map.get_mut(&k) {
                                    entry.status =
                                        AsyncRangeStatus::Ready(tracked.clone());
                                    if tracker.consume_key(k) {
                                        entry.remaining_consumers =
                                            entry.remaining_consumers.saturating_sub(1);
                                        if entry.remaining_consumers == 0
                                            && let Some(removed) = map.remove(&k)
                                        {
                                            evicted.push(removed);
                                        }
                                    }
                                }
                                let _ = tx.send(true);
                            }
                        }
                        drop(evicted);
                        break;
                    }
                }
            }
        }
        self.split_context
            .metrics
            .record_reused_bytes(waited_reused);

        let mut output = Vec::with_capacity(requested.len());
        for r in requested {
            let key = (r.start, r.end);
            let view = resolved.get(&key).ok_or_else(|| {
                internal_datafusion_err!(
                    "Missing resolved byte range {}..{} ({err_ctx})",
                    r.start,
                    r.end
                )
            })?;
            output.push(view.clone());
        }
        Ok(output)
    }
}

struct Phase2SliceTracker {
    broker: Arc<AsyncRangeBroker>,
    remaining_keys: HashSet<(u64, u64)>,
}

impl Phase2SliceTracker {
    fn new(broker: Arc<AsyncRangeBroker>, keys: Vec<(u64, u64)>) -> Self {
        Self {
            broker,
            remaining_keys: keys.into_iter().collect(),
        }
    }

    fn consume_key(&mut self, key: (u64, u64)) -> bool {
        self.remaining_keys.remove(&key)
    }
}

impl Drop for Phase2SliceTracker {
    fn drop(&mut self) {
        self.broker.release_unconsumed(&mut self.remaining_keys);
    }
}

pub(super) enum BufferedReaders {
    Plain(PlainBufferedReaders),
    SplitPlain(Box<SplitPlainBufferedReaders>),
    Filtered(Box<FilteredBufferedReaders>),
}

pub(super) struct PlainBufferedReaders {
    pub(super) decoder: ParquetPushDecoder,
    pub(super) reader: Box<dyn AsyncFileReader>,
    pub(super) projection: Arc<DecoderProjection>,
    pub(super) file_name: Arc<str>,
    pub(super) baseline: BaselineMetrics,
    pub(super) rg_plan: VecDeque<RgPlanEntry>,
    pub(super) byte_progress: ByteProgress,
    pub(super) ranges: Option<(Vec<Range<u64>>, Vec<Bytes>)>,
}

pub(super) struct SplitPlainBufferedReadersConfig {
    pub(super) pending: Vec<(RowGroupSelection, RgPlanEntry)>,
    pub(super) reader_metadata: ArrowReaderMetadata,
    pub(super) projection: Arc<DecoderProjection>,
    pub(super) batch_size: usize,
    pub(super) force_filter_selections: bool,
    pub(super) enable_page_index: bool,
    pub(super) arrow_reader_metrics: ArrowReaderMetrics,
    pub(super) file_name: Arc<str>,
    pub(super) baseline: BaselineMetrics,
    pub(super) byte_progress: ByteProgress,
    pub(super) reader: Box<dyn AsyncFileReader>,
    pub(super) split_context: Arc<SplitContext>,
}

enum PlainRgWork {
    Unsplit {
        selection: RowGroupSelection,
        entry: RgPlanEntry,
    },
    Split {
        rg_index: usize,
        original_selection: RowGroupSelection,
        entry: RgPlanEntry,
        plan: RowGroupSplitPlan,
    },
}

struct InFlightPlainSlice {
    slice_idx: usize,
    row_range: Range<usize>,
    selected_rows: usize,
    decoder: ParquetPushDecoder,
    requested_ranges: Vec<Range<u64>>,
    resolved_bytes: Vec<Bytes>,
    decoded_lease: SplitDecodedLease,
    compressed_bytes: usize,
}

struct ActiveSplitPlainRg {
    rg_index: usize,
    original_selection: RowGroupSelection,
    entry: RgPlanEntry,
    plan: RowGroupSplitPlan,
    slices_published: usize,
    frontier_guard: Option<FrontierGuard>,
    broker: RangeBroker,
    in_flight_slice: Option<InFlightPlainSlice>,
}

enum ActivePlainRg {
    Unsplit {
        decoder: ParquetPushDecoder,
        entry: RgPlanEntry,
        fetched: Option<(Vec<Range<u64>>, Vec<Bytes>)>,
        fetched_bytes: usize,
    },
    Split(Box<ActiveSplitPlainRg>),
}

pub(super) struct SplitPlainBufferedReaders {
    pending: VecDeque<PlainRgWork>,
    active: Option<ActivePlainRg>,
    reader_metadata: ArrowReaderMetadata,
    projection: Arc<DecoderProjection>,
    batch_size: usize,
    force_filter_selections: bool,
    arrow_reader_metrics: ArrowReaderMetrics,
    file_name: Arc<str>,
    baseline: BaselineMetrics,
    byte_progress: ByteProgress,
    reader: Box<dyn AsyncFileReader>,
    split_context: Arc<SplitContext>,
}

pub(super) struct FilteredBufferedReadersConfig {
    pub(super) pending: VecDeque<(RowGroupSelection, RgPlanEntry)>,
    pub(super) reader_metadata: ArrowReaderMetadata,
    pub(super) projection: Arc<DecoderProjection>,
    pub(super) bulk_projection: Option<Arc<DecoderProjection>>,
    pub(super) row_filter_context: RowFilterContext,
    pub(super) row_filter_skipped_fully_matched: RowFilterSkippedFullyMatchedMetric,
    pub(super) batch_size: usize,
    pub(super) force_filter_selections: bool,
    pub(super) enable_page_index: bool,
    pub(super) max_predicate_cache_size: Option<usize>,
    pub(super) arrow_reader_metrics: ArrowReaderMetrics,
    pub(super) file_metrics: ParquetFileMetrics,
    pub(super) file_name: Arc<str>,
    pub(super) file_location: Path,
    pub(super) file_size: u64,
    pub(super) baseline: BaselineMetrics,
    pub(super) byte_progress: ByteProgress,
    pub(super) async_file_reader: Box<dyn AsyncFileReader>,
    pub(super) direct_store: Option<Arc<dyn ObjectStore>>,
    pub(super) split_context: Option<Arc<SplitContext>>,
    pub(super) adaptive_context: Option<Arc<AdaptiveRowFilterContext>>,
}

type HandbackRequest = (Vec<Range<u64>>, oneshot::Sender<Result<Vec<Bytes>>>);

enum FilteredRgWork {
    Unsplit {
        selection: RowGroupSelection,
        entry: RgPlanEntry,
    },
    Split {
        rg_index: usize,
        original_selection: RowGroupSelection,
        entry: RgPlanEntry,
        plan: RowGroupSplitPlan,
    },
}

struct ActiveSplitFilteredRg {
    rg_index: usize,
    original_selection: RowGroupSelection,
    entry: RgPlanEntry,
    plan: RowGroupSplitPlan,
    slices_published: usize,
    frontier_guard: Option<FrontierGuard>,
    broker: Arc<AsyncRangeBroker>,
    split_context: Arc<SplitContext>,
}

pub(super) struct FilteredBufferedReaders {
    pending: VecDeque<FilteredRgWork>,
    file_surviving_rgs: usize,
    active_split: Option<Box<ActiveSplitFilteredRg>>,
    reader_metadata: ArrowReaderMetadata,
    projection: Arc<DecoderProjection>,
    bulk_projection: Option<Arc<DecoderProjection>>,
    row_filter_context: RowFilterContext,
    row_filter_skipped_fully_matched: RowFilterSkippedFullyMatchedMetric,
    batch_size: usize,
    force_filter_selections: bool,
    max_predicate_cache_size: Option<usize>,
    arrow_reader_metrics: ArrowReaderMetrics,
    predicate_cache_inner_records: Gauge,
    predicate_cache_records: Gauge,
    file_name: Arc<str>,
    baseline: BaselineMetrics,
    byte_progress: ByteProgress,
    /// Exclusively owned by this file's I/O lane (never shared behind a lock).
    reader: Box<dyn AsyncFileReader>,
    /// Lock-free object-store fetcher for Phase-2 I/O lanes when using a
    /// built-in reader factory; `None` for custom factories (which serve
    /// Phase-2 requests via `handback_rx` on this file's I/O lane).
    direct_fetcher: Option<Arc<DirectRangeFetcher>>,
    in_flight: Option<InFlight>,
    handback_rx: Option<mpsc::UnboundedReceiver<HandbackRequest>>,
    split_context: Option<Arc<SplitContext>>,
    adaptive_context: Option<Arc<AdaptiveRowFilterContext>>,
}

impl BufferedReaders {
    /// Plan splits for a plain (non-RowFilter) buffered scan. Returns
    /// `Ok(Some(SplitPlain))` when at least one row group splits, or `Ok(None)`
    /// (returning the original `pending` items untouched) so the caller can use
    /// the zero-overhead single-decoder [`PlainBufferedReaders`] fast path.
    pub(super) fn try_split_plain(
        config: SplitPlainBufferedReadersConfig,
    ) -> std::result::Result<Self, Box<SplitPlainBufferedReadersConfig>> {
        if !config.split_context.is_enabled() {
            if !config.pending.is_empty() {
                for _ in 0..config.pending.len() {
                    config
                        .split_context
                        .metrics
                        .record_unsplit_rg(UnsplitReason::Disabled);
                }
            }
            return Err(Box::new(config));
        }
        let metadata = config.reader_metadata.metadata();
        let proj_mask = config.projection.projection_mask();
        let mut work = VecDeque::with_capacity(config.pending.len());
        let mut any_split = false;

        for (selection, entry) in &config.pending {
            match plan_row_group_splits(
                metadata,
                entry.rg_index,
                selection.selection(),
                |leaf_idx| proj_mask.leaf_included(leaf_idx),
                |_| false,
                &config.split_context.config,
                config.enable_page_index,
                config.batch_size,
                config.split_context.budget.pool_limit(),
            ) {
                Ok(plan) => {
                    any_split = true;
                    work.push_back(PlainRgWork::Split {
                        rg_index: entry.rg_index,
                        original_selection: selection.clone(),
                        entry: entry.clone(),
                        plan,
                    });
                }
                Err(reason) => {
                    config.split_context.metrics.record_unsplit_rg(reason);
                    work.push_back(PlainRgWork::Unsplit {
                        selection: selection.clone(),
                        entry: entry.clone(),
                    });
                }
            }
        }

        if !any_split {
            return Err(Box::new(config));
        }

        Ok(Self::SplitPlain(Box::new(SplitPlainBufferedReaders {
            pending: work,
            active: None,
            reader_metadata: config.reader_metadata,
            projection: config.projection,
            batch_size: config.batch_size,
            force_filter_selections: config.force_filter_selections,
            arrow_reader_metrics: config.arrow_reader_metrics,
            file_name: config.file_name,
            baseline: config.baseline,
            byte_progress: config.byte_progress,
            reader: config.reader,
            split_context: config.split_context,
        })))
    }

    pub(super) fn filtered(config: FilteredBufferedReadersConfig) -> Self {
        let predicate_cache_inner_records =
            config.file_metrics.predicate_cache_inner_records.clone();
        let predicate_cache_records = config.file_metrics.predicate_cache_records.clone();
        let direct_fetcher = config.direct_store.map(|store| {
            Arc::new(DirectRangeFetcher {
                store,
                location: config.file_location,
                file_size: config.file_size,
                file_metrics: config.file_metrics,
            })
        });

        let metadata = config.reader_metadata.metadata();
        let proj_mask = config.projection.projection_mask();
        let pending: VecDeque<FilteredRgWork> = config
            .pending
            .into_iter()
            .map(|(selection, entry)| {
                let Some(split_ctx) = &config.split_context else {
                    return FilteredRgWork::Unsplit { selection, entry };
                };
                let multi_pred = config.row_filter_context.num_predicates() > 1;
                match plan_row_group_splits(
                    metadata,
                    entry.rg_index,
                    selection.selection(),
                    |leaf_idx| {
                        proj_mask.leaf_included(leaf_idx)
                            || config.row_filter_context.reads_leaf(leaf_idx)
                    },
                    |leaf_idx| {
                        !entry.fully_matched
                            && config.row_filter_context.reads_leaf(leaf_idx)
                            && (multi_pred || proj_mask.leaf_included(leaf_idx))
                    },
                    &split_ctx.config,
                    config.enable_page_index,
                    config.batch_size,
                    split_ctx.budget.pool_limit(),
                ) {
                    Ok(plan) => FilteredRgWork::Split {
                        rg_index: entry.rg_index,
                        original_selection: selection,
                        entry,
                        plan,
                    },
                    Err(reason) => {
                        split_ctx.metrics.record_unsplit_rg(reason);
                        FilteredRgWork::Unsplit { selection, entry }
                    }
                }
            })
            .collect();

        let file_surviving_rgs = pending.len();
        Self::Filtered(Box::new(FilteredBufferedReaders {
            pending,
            file_surviving_rgs,
            active_split: None,
            reader_metadata: config.reader_metadata,
            projection: config.projection,
            bulk_projection: config.bulk_projection,
            row_filter_context: config.row_filter_context,
            row_filter_skipped_fully_matched: config.row_filter_skipped_fully_matched,
            batch_size: config.batch_size,
            force_filter_selections: config.force_filter_selections,
            max_predicate_cache_size: config.max_predicate_cache_size,
            arrow_reader_metrics: config.arrow_reader_metrics,
            predicate_cache_inner_records,
            predicate_cache_records,
            file_name: config.file_name,
            baseline: config.baseline,
            byte_progress: config.byte_progress,
            reader: config.async_file_reader,
            direct_fetcher,
            in_flight: None,
            handback_rx: None,
            split_context: config.split_context,
            adaptive_context: config.adaptive_context,
        }))
    }
}

impl fmt::Debug for BufferedReaders {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BufferedReaders").finish_non_exhaustive()
    }
}

impl MorselPlanner for BufferedReaders {
    fn plan(self: Box<Self>) -> Result<Option<MorselPlan>> {
        match *self {
            Self::Plain(plain) => plain.plan(),
            Self::SplitPlain(split) => split.plan(),
            Self::Filtered(filtered) => filtered.plan(),
        }
    }
}

impl PlainBufferedReaders {
    fn plan(mut self) -> Result<Option<MorselPlan>> {
        let elapsed_compute = self.baseline.elapsed_compute().clone();
        let _timer = elapsed_compute.timer();
        if let Some((ranges, bytes)) = self.ranges.take() {
            self.decoder.push_ranges(ranges, bytes)?;
        }
        let plan = match self.decoder.try_next_reader()? {
            DecodeResult::NeedsData(ranges) => {
                MorselPlan::new().with_pending_planner(async move {
                    let bytes = self.reader.get_byte_ranges(ranges.clone()).await?;
                    self.ranges = Some((ranges, bytes));
                    Ok(Box::new(BufferedReaders::Plain(self)) as Box<dyn MorselPlanner>)
                })
            }
            DecodeResult::Data(reader) => {
                // After extraction the decoder is at a boundary. It may have
                // skipped empty selections before arriving at this reader.
                let remaining = self.decoder.row_groups_remaining();
                let mut compressed_bytes = 0;
                while self.rg_plan.len() > remaining {
                    compressed_bytes = self.rg_plan.pop_front().unwrap().bytes;
                    self.byte_progress.credit(compressed_bytes);
                }
                let reader = BufferedReader {
                    reader,
                    compressed_bytes,
                    projection: Arc::clone(&self.projection),
                    file_name: Arc::clone(&self.file_name),
                    row_context: None,
                    elapsed_compute: elapsed_compute.clone(),
                    cache_metrics: None,
                    decoded_lease: None,
                    in_flight_guard: None,
                    rg_handle: None,
                };
                MorselPlan::new()
                    .with_morsels(vec![Box::new(reader)])
                    .with_planners(vec![Box::new(BufferedReaders::Plain(self))])
            }
            DecodeResult::Finished => return Ok(None),
        };
        Ok(Some(plan))
    }
}

impl SplitPlainBufferedReaders {
    fn build_decoder(&self, selection: RowGroupSelection) -> Result<ParquetPushDecoder> {
        let mut builder =
            ParquetPushDecoderBuilder::new_with_metadata(self.reader_metadata.clone())
                .with_projection(self.projection.projection_mask().clone())
                .with_batch_size(self.batch_size)
                .with_metrics(self.arrow_reader_metrics.clone())
                .with_row_group_selections(vec![selection]);
        if self.force_filter_selections {
            builder = builder.with_row_selection_policy(RowSelectionPolicy::Selectors);
        }
        Ok(builder.build()?)
    }

    fn slice_context(&self, rg_index: usize, row_range: &Range<usize>) -> String {
        format!(
            "file={}, row_group={rg_index}, rows={}..{}",
            self.file_name, row_range.start, row_range.end
        )
    }

    fn publish_morsel(self: Box<Self>, morsel: Box<dyn Morsel>) -> MorselPlan {
        let mut plan = MorselPlan::new().with_morsels(vec![morsel]);
        if self.active.is_some() || !self.pending.is_empty() {
            plan = plan.with_planners(vec![Box::new(BufferedReaders::SplitPlain(self))]);
        }
        plan
    }

    /// Before any slice of the active split row group has been published, fall
    /// back cleanly to the unsplit reader instead of failing on memory pressure.
    fn fallback_active_split_to_unsplit(&mut self) -> Result<()> {
        let Some(ActivePlainRg::Split(rg)) = self.active.take() else {
            return Ok(());
        };
        debug_assert_eq!(rg.slices_published, 0);
        let original_selection = rg.original_selection;
        let entry = rg.entry;
        // Drop frontier_guard & broker before switching to unsplit.
        drop(rg.frontier_guard);
        drop(rg.broker);
        self.split_context
            .metrics
            .record_unsplit_rg(UnsplitReason::MemoryPoolCap);
        let decoder = self.build_decoder(original_selection)?;
        self.active = Some(ActivePlainRg::Unsplit {
            decoder,
            entry,
            fetched: None,
            fetched_bytes: 0,
        });
        Ok(())
    }

    fn plan(mut self: Box<Self>) -> Result<Option<MorselPlan>> {
        let elapsed_compute = self.baseline.elapsed_compute().clone();
        let _timer = elapsed_compute.timer();

        loop {
            if self.active.is_none() {
                let Some(next) = self.pending.pop_front() else {
                    return Ok(None);
                };
                match next {
                    PlainRgWork::Unsplit { selection, entry } => {
                        let decoder = self.build_decoder(selection)?;
                        self.active = Some(ActivePlainRg::Unsplit {
                            decoder,
                            entry,
                            fetched: None,
                            fetched_bytes: 0,
                        });
                    }
                    PlainRgWork::Split {
                        rg_index,
                        original_selection,
                        entry,
                        plan,
                    } => {
                        let frontier_guard =
                            self.split_context.budget.try_acquire_frontier(
                                plan.frontier_bytes,
                                plan.pool_frontier_bytes,
                            );
                        self.active =
                            Some(ActivePlainRg::Split(Box::new(ActiveSplitPlainRg {
                                rg_index,
                                original_selection,
                                entry,
                                plan,
                                slices_published: 0,
                                frontier_guard,
                                broker: RangeBroker::new(),
                                in_flight_slice: None,
                            })));
                    }
                }
            }

            match self.active.take().unwrap() {
                ActivePlainRg::Unsplit {
                    mut decoder,
                    entry,
                    mut fetched,
                    mut fetched_bytes,
                } => {
                    if let Some((ranges, bytes)) = fetched.take() {
                        fetched_bytes += bytes.iter().map(Bytes::len).sum::<usize>();
                        decoder.push_ranges(ranges, bytes)?;
                    }
                    match decoder.try_next_reader()? {
                        DecodeResult::NeedsData(ranges) => {
                            self.active = Some(ActivePlainRg::Unsplit {
                                decoder,
                                entry,
                                fetched: None,
                                fetched_bytes,
                            });
                            return Ok(Some(MorselPlan::new().with_pending_planner(
                                async move {
                                    let bytes = self
                                        .reader
                                        .get_byte_ranges(ranges.clone())
                                        .await?;
                                    if let Some(ActivePlainRg::Unsplit {
                                        fetched, ..
                                    }) = self.active.as_mut()
                                    {
                                        *fetched = Some((ranges, bytes));
                                    }
                                    Ok(Box::new(BufferedReaders::SplitPlain(self))
                                        as Box<dyn MorselPlanner>)
                                },
                            )));
                        }
                        DecodeResult::Data(reader) => {
                            self.byte_progress.credit(entry.bytes);
                            let morsel = Box::new(BufferedReader {
                                reader,
                                compressed_bytes: fetched_bytes as u64,
                                projection: Arc::clone(&self.projection),
                                file_name: Arc::clone(&self.file_name),
                                row_context: None,
                                elapsed_compute: elapsed_compute.clone(),
                                cache_metrics: None,
                                decoded_lease: None,
                                in_flight_guard: None,
                                rg_handle: None,
                            });
                            return Ok(Some(self.publish_morsel(morsel)));
                        }
                        DecodeResult::Finished => {
                            self.byte_progress.credit(entry.bytes);
                        }
                    }
                }
                ActivePlainRg::Split(mut rg) => {
                    if rg.frontier_guard.is_none() {
                        let frontier_bytes = rg.plan.frontier_bytes;
                        let pool_frontier_bytes = rg.plan.pool_frontier_bytes;
                        let is_first_slice = rg.slices_published == 0;
                        let budget = Arc::clone(&self.split_context.budget);
                        self.active = Some(ActivePlainRg::Split(rg));
                        return Ok(Some(MorselPlan::new().with_pending_planner(
                            async move {
                                match budget
                                    .acquire_frontier(frontier_bytes, pool_frontier_bytes)
                                    .await
                                {
                                    Ok(guard) => {
                                        if let Some(ActivePlainRg::Split(rg)) =
                                            self.active.as_mut()
                                        {
                                            rg.frontier_guard = Some(guard);
                                        }
                                    }
                                    Err(DataFusionError::ResourcesExhausted(_))
                                        if is_first_slice =>
                                    {
                                        self.fallback_active_split_to_unsplit()?;
                                    }
                                    Err(err) => return Err(err),
                                }
                                Ok(Box::new(BufferedReaders::SplitPlain(self))
                                    as Box<dyn MorselPlanner>)
                            },
                        )));
                    }

                    if let Some(in_flight) = rg.in_flight_slice.take() {
                        let InFlightPlainSlice {
                            slice_idx,
                            row_range,
                            selected_rows,
                            mut decoder,
                            requested_ranges,
                            resolved_bytes,
                            decoded_lease,
                            compressed_bytes,
                        } = in_flight;
                        let ctx = self.slice_context(rg.rg_index, &row_range);
                        if !requested_ranges.is_empty() {
                            decoder
                                .push_ranges(requested_ranges, resolved_bytes)
                                .map_err(|e| {
                                    DataFusionError::from(e).context(format!(
                                        "Pushing Parquet split ranges ({ctx})"
                                    ))
                                })?;
                        }
                        match decoder.try_next_reader().map_err(|e| {
                            DataFusionError::from(e).context(format!(
                                "Preparing Parquet split reader ({ctx})"
                            ))
                        })? {
                            DecodeResult::NeedsData(more_ranges) => {
                                // Rare multi-step plain decode: fetch additional ranges
                                // while retaining the already-acquired decoded lease.
                                let rg_index = rg.rg_index;
                                let misses = rg.broker.partition_misses(&more_ranges);
                                let coalesced = coalesce_misses(
                                    misses,
                                    &rg.plan.last_consumer,
                                    slice_idx,
                                    self.split_context.config.max_coalesced_request_bytes,
                                );
                                let (lease_lengths, has_ephemeral) =
                                    plan_range_broker_leases(
                                        &coalesced,
                                        &rg.plan.last_consumer,
                                        slice_idx,
                                    );
                                self.active = Some(ActivePlainRg::Split(rg));
                                return Ok(Some(MorselPlan::new().with_pending_planner(
                                    async move {
                                        let (leases, _, _) = self
                                            .split_context
                                            .budget
                                            .reserve_for_fetch(&lease_lengths, || 0, 0)
                                            .await?;
                                        let fetch_ranges: Vec<Range<u64>> = coalesced
                                            .iter()
                                            .map(|c| c.range.clone())
                                            .collect();
                                        let raw = if fetch_ranges.is_empty() {
                                            Vec::new()
                                        } else {
                                            self.reader
                                                .get_byte_ranges(fetch_ranges)
                                                .await
                                                .map_err(|e| {
                                                    DataFusionError::from(e).context(
                                                        format!(
                                                            "Reading Parquet split morsel ({ctx})"
                                                        ),
                                                    )
                                                })?
                                        };
                                        let resolved = if let Some(
                                            ActivePlainRg::Split(rg),
                                        ) = self.active.as_mut()
                                        {
                                            rg.broker.complete_fetch(
                                                &more_ranges,
                                                &coalesced,
                                                raw,
                                                leases,
                                                has_ephemeral,
                                                &rg.plan.last_consumer,
                                                slice_idx,
                                                &self.split_context.metrics,
                                                &ctx,
                                            )?
                                        } else {
                                            unreachable!()
                                        };
                                        let added: usize =
                                            resolved.iter().map(Bytes::len).sum();
                                        if let Some(ActivePlainRg::Split(rg)) =
                                            self.active.as_mut()
                                        {
                                            debug_assert_eq!(rg.rg_index, rg_index);
                                            rg.in_flight_slice =
                                                Some(InFlightPlainSlice {
                                                    slice_idx,
                                                    row_range,
                                                    selected_rows,
                                                    decoder,
                                                    requested_ranges: more_ranges,
                                                    resolved_bytes: resolved,
                                                    decoded_lease,
                                                    compressed_bytes: compressed_bytes
                                                        + added,
                                                });
                                        }
                                        Ok(Box::new(BufferedReaders::SplitPlain(self))
                                            as Box<dyn MorselPlanner>)
                                    },
                                )));
                            }
                            DecodeResult::Data(reader) => {
                                if rg.slices_published == 0 {
                                    self.split_context.metrics.record_split_rg();
                                }
                                rg.slices_published += 1;
                                self.split_context.metrics.record_slice(selected_rows);
                                let rg_index = rg.rg_index;
                                // Register the decode reader BEFORE dropping `rg`
                                // (and `rg.frontier_guard`) so waiting lanes never
                                // observe zero active lanes + zero in-flight readers
                                // while this reader's compressed bytes are still live.
                                let in_flight_guard =
                                    self.split_context.budget.track_decode_reader();
                                if rg.plan.slices.is_empty() {
                                    self.byte_progress.credit(rg.entry.bytes);
                                    drop(rg);
                                } else {
                                    self.active = Some(ActivePlainRg::Split(rg));
                                }
                                let morsel = Box::new(BufferedReader {
                                    reader,
                                    compressed_bytes: compressed_bytes as u64,
                                    projection: Arc::clone(&self.projection),
                                    file_name: Arc::clone(&self.file_name),
                                    row_context: Some((rg_index, row_range)),
                                    elapsed_compute: elapsed_compute.clone(),
                                    cache_metrics: None,
                                    decoded_lease: Some(decoded_lease),
                                    in_flight_guard: Some(in_flight_guard),
                                    rg_handle: None,
                                });
                                return Ok(Some(self.publish_morsel(morsel)));
                            }
                            DecodeResult::Finished => {
                                if rg.plan.slices.is_empty() {
                                    self.byte_progress.credit(rg.entry.bytes);
                                } else {
                                    self.active = Some(ActivePlainRg::Split(rg));
                                }
                                continue;
                            }
                        }
                    }

                    let Some(slice) = rg.plan.slices.pop_front() else {
                        self.byte_progress.credit(rg.entry.bytes);
                        continue;
                    };
                    let ctx = self.slice_context(rg.rg_index, &slice.row_range);
                    let mut decoder = self
                        .build_decoder(RowGroupSelection::new(
                            rg.rg_index,
                            Some(slice.selection),
                        ))
                        .map_err(|e| {
                            e.context(format!("Building Parquet split decoder ({ctx})"))
                        })?;

                    match decoder.try_next_reader().map_err(|e| {
                        DataFusionError::from(e)
                            .context(format!("Preparing Parquet split reader ({ctx})"))
                    })? {
                        DecodeResult::NeedsData(requested_ranges) => {
                            let misses = rg.broker.partition_misses(&requested_ranges);
                            let decoded_estimate = slice.decoded_estimate;
                            let is_first_slice = rg.slices_published == 0;
                            if misses.is_empty() {
                                if let Some(decoded_lease) = self
                                    .split_context
                                    .budget
                                    .try_reserve_decoded_sync(decoded_estimate)
                                {
                                    let resolved_bytes = rg.broker.serve_hits_only(
                                        &requested_ranges,
                                        slice.slice_idx,
                                        &self.split_context.metrics,
                                        &ctx,
                                    )?;
                                    let compressed_bytes: usize =
                                        resolved_bytes.iter().map(Bytes::len).sum();
                                    rg.in_flight_slice = Some(InFlightPlainSlice {
                                        slice_idx: slice.slice_idx,
                                        row_range: slice.row_range,
                                        selected_rows: slice.selected_rows,
                                        decoder,
                                        requested_ranges,
                                        resolved_bytes,
                                        decoded_lease,
                                        compressed_bytes,
                                    });
                                    self.active = Some(ActivePlainRg::Split(rg));
                                    continue;
                                }
                                self.active = Some(ActivePlainRg::Split(rg));
                                return Ok(Some(MorselPlan::new().with_pending_planner(
                                    async move {
                                        let decoded_lease = match self
                                            .split_context
                                            .budget
                                            .reserve_decoded(decoded_estimate)
                                            .await
                                        {
                                            Ok(lease) => lease,
                                            Err(DataFusionError::ResourcesExhausted(
                                                _,
                                            )) if is_first_slice => {
                                                self.fallback_active_split_to_unsplit()?;
                                                return Ok(Box::new(
                                                    BufferedReaders::SplitPlain(self),
                                                )
                                                    as Box<dyn MorselPlanner>);
                                            }
                                            Err(err) => return Err(err),
                                        };
                                        if let Some(ActivePlainRg::Split(rg)) =
                                            self.active.as_mut()
                                        {
                                            let resolved_bytes =
                                                rg.broker.serve_hits_only(
                                                    &requested_ranges,
                                                    slice.slice_idx,
                                                    &self.split_context.metrics,
                                                    &ctx,
                                                )?;
                                            let compressed_bytes: usize = resolved_bytes
                                                .iter()
                                                .map(Bytes::len)
                                                .sum();
                                            rg.in_flight_slice =
                                                Some(InFlightPlainSlice {
                                                    slice_idx: slice.slice_idx,
                                                    row_range: slice.row_range,
                                                    selected_rows: slice.selected_rows,
                                                    decoder,
                                                    requested_ranges,
                                                    resolved_bytes,
                                                    decoded_lease,
                                                    compressed_bytes,
                                                });
                                        }
                                        Ok(Box::new(BufferedReaders::SplitPlain(self))
                                            as Box<dyn MorselPlanner>)
                                    },
                                )));
                            }

                            let coalesced = coalesce_misses(
                                misses,
                                &rg.plan.last_consumer,
                                slice.slice_idx,
                                self.split_context.config.max_coalesced_request_bytes,
                            );
                            let (lease_lengths, has_ephemeral) = plan_range_broker_leases(
                                &coalesced,
                                &rg.plan.last_consumer,
                                slice.slice_idx,
                            );
                            let fetch_ranges: Vec<Range<u64>> =
                                coalesced.iter().map(|c| c.range.clone()).collect();
                            self.active = Some(ActivePlainRg::Split(rg));
                            return Ok(Some(MorselPlan::new().with_pending_planner(
                                async move {
                                    let (leases, _phase2, decoded_lease) = match self
                                        .split_context
                                        .budget
                                        .reserve_for_fetch(
                                            &lease_lengths,
                                            || 0,
                                            decoded_estimate,
                                        )
                                        .await
                                    {
                                        Ok(res) => res,
                                        Err(DataFusionError::ResourcesExhausted(_))
                                            if is_first_slice =>
                                        {
                                            self.fallback_active_split_to_unsplit()?;
                                            return Ok(Box::new(
                                                BufferedReaders::SplitPlain(self),
                                            )
                                                as Box<dyn MorselPlanner>);
                                        }
                                        Err(err) => return Err(err),
                                    };
                                    let raw = self
                                        .reader
                                        .get_byte_ranges(fetch_ranges)
                                        .await
                                        .map_err(|e| {
                                            DataFusionError::from(e).context(format!(
                                                "Reading Parquet split morsel ({ctx})"
                                            ))
                                        })?;
                                    if let Some(ActivePlainRg::Split(rg)) =
                                        self.active.as_mut()
                                    {
                                        let resolved_bytes = rg.broker.complete_fetch(
                                            &requested_ranges,
                                            &coalesced,
                                            raw,
                                            leases,
                                            has_ephemeral,
                                            &rg.plan.last_consumer,
                                            slice.slice_idx,
                                            &self.split_context.metrics,
                                            &ctx,
                                        )?;
                                        let compressed_bytes: usize =
                                            resolved_bytes.iter().map(Bytes::len).sum();
                                        rg.in_flight_slice = Some(InFlightPlainSlice {
                                            slice_idx: slice.slice_idx,
                                            row_range: slice.row_range,
                                            selected_rows: slice.selected_rows,
                                            decoder,
                                            requested_ranges,
                                            resolved_bytes,
                                            decoded_lease,
                                            compressed_bytes,
                                        });
                                    }
                                    Ok(Box::new(BufferedReaders::SplitPlain(self))
                                        as Box<dyn MorselPlanner>)
                                },
                            )));
                        }
                        DecodeResult::Data(reader) => {
                            let decoded_estimate = slice.decoded_estimate;
                            let is_first_slice = rg.slices_published == 0;
                            let decoded_lease = self
                                .split_context
                                .budget
                                .try_reserve_decoded_sync(decoded_estimate);
                            let Some(decoded_lease) = decoded_lease else {
                                let rg_index = rg.rg_index;
                                let row_range = slice.row_range;
                                let selected_rows = slice.selected_rows;
                                let last_slice = rg.plan.slices.is_empty();
                                let entry_bytes = rg.entry.bytes;
                                // Keep `rg` (and `frontier_guard`) alive across the
                                // reservation wait so slice 0 can fall back cleanly.
                                self.active = Some(ActivePlainRg::Split(rg));
                                let projection = Arc::clone(&self.projection);
                                let file_name = Arc::clone(&self.file_name);
                                let split_context = Arc::clone(&self.split_context);
                                let elapsed_compute_owned = elapsed_compute.clone();
                                return Ok(Some(MorselPlan::new().with_pending_planner(
                                    async move {
                                        let decoded_lease = match split_context
                                            .budget
                                            .reserve_decoded(decoded_estimate)
                                            .await
                                        {
                                            Ok(lease) => lease,
                                            Err(DataFusionError::ResourcesExhausted(
                                                _,
                                            )) if is_first_slice => {
                                                self.fallback_active_split_to_unsplit()?;
                                                return Ok(Box::new(
                                                    BufferedReaders::SplitPlain(self),
                                                )
                                                    as Box<dyn MorselPlanner>);
                                            }
                                            Err(err) => return Err(err),
                                        };
                                        if is_first_slice {
                                            split_context.metrics.record_split_rg();
                                        }
                                        split_context.metrics.record_slice(selected_rows);
                                        let in_flight_guard =
                                            split_context.budget.track_decode_reader();
                                        if last_slice {
                                            self.byte_progress.credit(entry_bytes);
                                            self.active = None;
                                        } else if let Some(ActivePlainRg::Split(rg)) =
                                            self.active.as_mut()
                                        {
                                            rg.slices_published += 1;
                                        }
                                        let morsel = Box::new(BufferedReader {
                                            reader,
                                            compressed_bytes: 0,
                                            projection,
                                            file_name,
                                            row_context: Some((rg_index, row_range)),
                                            elapsed_compute: elapsed_compute_owned,
                                            cache_metrics: None,
                                            decoded_lease: Some(decoded_lease),
                                            in_flight_guard: Some(in_flight_guard),
                                            rg_handle: None,
                                        });
                                        Ok(Box::new(PublishReadyMorsel {
                                            morsel: Some(morsel),
                                            continuation: if self.active.is_some()
                                                || !self.pending.is_empty()
                                            {
                                                Some(Box::new(
                                                    BufferedReaders::SplitPlain(self),
                                                ))
                                            } else {
                                                None
                                            },
                                        })
                                            as Box<dyn MorselPlanner>)
                                    },
                                )));
                            };
                            if rg.slices_published == 0 {
                                self.split_context.metrics.record_split_rg();
                            }
                            rg.slices_published += 1;
                            self.split_context.metrics.record_slice(slice.selected_rows);
                            let rg_index = rg.rg_index;
                            let in_flight_guard =
                                self.split_context.budget.track_decode_reader();
                            if rg.plan.slices.is_empty() {
                                self.byte_progress.credit(rg.entry.bytes);
                                drop(rg);
                            } else {
                                self.active = Some(ActivePlainRg::Split(rg));
                            }
                            let morsel = Box::new(BufferedReader {
                                reader,
                                compressed_bytes: 0,
                                projection: Arc::clone(&self.projection),
                                file_name: Arc::clone(&self.file_name),
                                row_context: Some((rg_index, slice.row_range)),
                                elapsed_compute: elapsed_compute.clone(),
                                cache_metrics: None,
                                decoded_lease: Some(decoded_lease),
                                in_flight_guard: Some(in_flight_guard),
                                rg_handle: None,
                            });
                            return Ok(Some(self.publish_morsel(morsel)));
                        }
                        DecodeResult::Finished => {
                            rg.broker.evict_through(slice.slice_idx);
                            if rg.plan.slices.is_empty() {
                                self.byte_progress.credit(rg.entry.bytes);
                            } else {
                                self.active = Some(ActivePlainRg::Split(rg));
                            }
                        }
                    }
                }
            }
        }
    }
}

struct PublishReadyMorsel {
    morsel: Option<Box<dyn Morsel>>,
    continuation: Option<Box<dyn MorselPlanner>>,
}

impl fmt::Debug for PublishReadyMorsel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PublishReadyMorsel").finish_non_exhaustive()
    }
}

impl MorselPlanner for PublishReadyMorsel {
    fn plan(mut self: Box<Self>) -> Result<Option<MorselPlan>> {
        let mut plan = MorselPlan::new();
        if let Some(morsel) = self.morsel.take() {
            plan = plan.with_morsels(vec![morsel]);
        }
        if let Some(cont) = self.continuation.take() {
            plan = plan.with_planners(vec![cont]);
        }
        Ok(Some(plan))
    }
}

struct DirectRangeFetcher {
    store: Arc<dyn ObjectStore>,
    location: Path,
    file_size: u64,
    file_metrics: ParquetFileMetrics,
}

impl DirectRangeFetcher {
    async fn get_byte_ranges(&self, ranges: Vec<Range<u64>>) -> Result<Vec<Bytes>> {
        let total: u64 = ranges.iter().map(|r| r.end - r.start).sum();
        self.file_metrics.bytes_scanned.add(total as usize);
        self.store
            .get_ranges(&self.location, &ranges)
            .await
            .map_err(|err| DataFusionError::from(ParquetError::External(Box::new(err))))
    }
}

impl Drop for DirectRangeFetcher {
    fn drop(&mut self) {
        self.file_metrics
            .scan_efficiency_ratio
            .add_part(self.file_metrics.bytes_scanned.value());
        self.file_metrics
            .scan_efficiency_ratio
            .set_total(self.file_size as usize);
    }
}

#[derive(Clone)]
struct PredicateCacheMetrics {
    arrow_reader_metrics: ArrowReaderMetrics,
    predicate_cache_inner_records: Gauge,
    predicate_cache_records: Gauge,
}

impl PredicateCacheMetrics {
    fn copy_to_gauges(&self) {
        if let Some(v) = self.arrow_reader_metrics.records_read_from_inner() {
            self.predicate_cache_inner_records.set(v);
        }
        if let Some(v) = self.arrow_reader_metrics.records_read_from_cache() {
            self.predicate_cache_records.set(v);
        }
    }
}

#[derive(Clone)]
enum Phase2Transport {
    Direct(Arc<DirectRangeFetcher>),
    Handback(mpsc::UnboundedSender<HandbackRequest>),
}

impl Phase2Transport {
    async fn get_byte_ranges(&self, ranges: Vec<Range<u64>>) -> Result<Vec<Bytes>> {
        match self {
            Self::Direct(fetcher) => fetcher.get_byte_ranges(ranges).await,
            Self::Handback(tx) => {
                let (reply_tx, reply_rx) = oneshot::channel();
                tx.send((ranges, reply_tx)).map_err(|_| {
                    internal_datafusion_err!(
                        "Parquet file I/O lane closed before serving Phase-2 range read"
                    )
                })?;
                reply_rx.await.map_err(|_| {
                    internal_datafusion_err!(
                        "Parquet file I/O lane dropped Phase-2 range response"
                    )
                })?
            }
        }
    }
}

struct SplitSliceMeta {
    rg_index: usize,
    row_range: Range<usize>,
    selected_rows: usize,
    is_last_slice: bool,
    credit_entry_bytes: Option<u64>,
    phase2_lease: Phase2BudgetLease,
    decoded_lease: SplitDecodedLease,
    tracker: Phase2SliceTracker,
    broker: Arc<AsyncRangeBroker>,
    split_context: Arc<SplitContext>,
}

enum InFlight {
    Filtered {
        decoder: ParquetPushDecoder,
        entry: RgPlanEntry,
        ranges: Vec<Range<u64>>,
        bytes: Vec<Bytes>,
        split_meta: Option<Box<SplitSliceMeta>>,
        rg_handle: Option<RgProbeHandle>,
    },
    FullyMatched {
        decoder: ParquetPushDecoder,
        entry: RgPlanEntry,
        ranges: Vec<Range<u64>>,
        bytes: Vec<Bytes>,
        fetched_bytes: usize,
        split_meta: Option<Box<SplitSliceMeta>>,
        use_bulk_projection: bool,
        rg_handle: Option<RgProbeHandle>,
    },
}

/// Sentinel file continuation emitted alongside a worker-bound
/// [`FilteredRowGroupPlanner`] when the file has no more row groups to open.
#[derive(Debug)]
struct EmptyDonePlanner;

impl MorselPlanner for EmptyDonePlanner {
    fn plan(self: Box<Self>) -> Result<Option<MorselPlan>> {
        Ok(None)
    }
}

impl FilteredBufferedReaders {
    fn can_row_filter_skip_columns(&self) -> bool {
        if self
            .adaptive_context
            .as_ref()
            .is_some_and(|ctx| ctx.has_dynamic_join_filter())
        {
            return true;
        }
        if self.row_filter_context.num_predicates() >= 2 {
            return true;
        }
        let proj_mask = self.projection.projection_mask();
        let num_leaves = self.reader_metadata.parquet_schema().num_columns();
        (0..num_leaves).any(|leaf_idx| {
            proj_mask.leaf_included(leaf_idx)
                && !self.row_filter_context.reads_leaf(leaf_idx)
        })
    }

    fn build_rg_decoder(
        &self,
        selection: RowGroupSelection,
        projection: &DecoderProjection,
        install_filter: bool,
        rg_handle: &mut Option<RgProbeHandle>,
    ) -> Result<ParquetPushDecoder> {
        let mut builder =
            ParquetPushDecoderBuilder::new_with_metadata(self.reader_metadata.clone())
                .with_projection(projection.projection_mask().clone())
                .with_batch_size(self.batch_size)
                .with_metrics(self.arrow_reader_metrics.clone())
                .with_row_group_selections(vec![selection]);
        if self.force_filter_selections {
            builder = builder.with_row_selection_policy(RowSelectionPolicy::Selectors);
        }
        if install_filter {
            let row_filter = if let Some(adaptive) = &self.adaptive_context {
                let (order, order_epoch) =
                    adaptive.row_filter_order(self.row_filter_context.candidates());
                if let Some(handle) = rg_handle.as_mut() {
                    handle.set_rf_order_epoch(order_epoch);
                }
                self.row_filter_context.build_row_filter_with_order(&order)
            } else {
                self.row_filter_context.build_row_filter()
            };
            builder = builder.with_row_filter(row_filter);
            if let Some(cap) = self.max_predicate_cache_size {
                builder = builder.with_max_predicate_cache_size(cap);
            }
        }
        Ok(builder.build()?)
    }

    fn cache_metrics(&self) -> PredicateCacheMetrics {
        PredicateCacheMetrics {
            arrow_reader_metrics: self.arrow_reader_metrics.clone(),
            predicate_cache_inner_records: self.predicate_cache_inner_records.clone(),
            predicate_cache_records: self.predicate_cache_records.clone(),
        }
    }

    fn has_more_work(&self) -> bool {
        self.handback_rx.is_some()
            || self.active_split.is_some()
            || !self.pending.is_empty()
    }

    fn publish_morsel(self: Box<Self>, morsel: Box<dyn Morsel>) -> MorselPlan {
        let mut plan = MorselPlan::new().with_morsels(vec![morsel]);
        if self.has_more_work() {
            plan = plan.with_planners(vec![Box::new(BufferedReaders::Filtered(self))]);
        }
        plan
    }

    /// Before any slice of the active split row group has been published, fall
    /// back cleanly to the unsplit reader instead of failing on memory pressure.
    fn fallback_active_split_to_unsplit(&mut self) {
        let Some(split_rg) = self.active_split.take() else {
            return;
        };
        debug_assert_eq!(split_rg.slices_published, 0);
        let original_selection = split_rg.original_selection;
        let entry = split_rg.entry;
        let split_context = split_rg.split_context;
        // Release frontier_guard and broker before re-queuing unsplit work.
        drop(split_rg.frontier_guard);
        drop(split_rg.broker);
        split_context
            .metrics
            .record_unsplit_rg(UnsplitReason::MemoryPoolCap);
        self.pending.push_front(FilteredRgWork::Unsplit {
            selection: original_selection,
            entry,
        });
    }

    /// Mark one slice of the active split row group as published and release
    /// `active_split` (including its `FrontierGuard`) once the last slice is
    /// registered in-flight.
    fn finish_published_split_slice(&mut self, is_last_slice: bool, fully_matched: bool) {
        if let Some(rg) = self.active_split.as_mut() {
            if rg.slices_published == 0 {
                rg.split_context.metrics.record_split_rg();
                if fully_matched {
                    self.row_filter_skipped_fully_matched.add_one();
                }
            }
            rg.slices_published += 1;
        }
        if is_last_slice {
            self.active_split = None;
        }
    }

    fn plan(mut self: Box<Self>) -> Result<Option<MorselPlan>> {
        let elapsed_compute = self.baseline.elapsed_compute().clone();
        let _timer = elapsed_compute.timer();

        // For custom reader factories, serve any Phase-2 range reads for the
        // previously published row group / slice on this file lane until it
        // finishes or publishes its decode-only reader (dropping its sender).
        if let Some(mut rx) = self.handback_rx.take() {
            return Ok(Some(MorselPlan::new().with_pending_planner(async move {
                while let Some((ranges, reply_tx)) = rx.recv().await {
                    let result = self
                        .reader
                        .get_byte_ranges(ranges)
                        .await
                        .map_err(DataFusionError::from);
                    let _ = reply_tx.send(result);
                }
                Ok(Box::new(BufferedReaders::Filtered(self)) as Box<dyn MorselPlanner>)
            })));
        }

        if let Some(in_flight) = self.in_flight.take() {
            match in_flight {
                InFlight::Filtered {
                    mut decoder,
                    entry,
                    ranges,
                    bytes,
                    split_meta,
                    mut rg_handle,
                } => {
                    let buffered_bytes = bytes.iter().map(Bytes::len).sum();
                    if let Some(handle) = rg_handle.as_mut() {
                        handle.add_bytes_fetched(buffered_bytes as u64);
                    }
                    if let Some(meta) = &split_meta {
                        let ctx = format!(
                            "file={}, row_group={}, rows={}..{}",
                            self.file_name,
                            meta.rg_index,
                            meta.row_range.start,
                            meta.row_range.end
                        );
                        decoder.push_ranges(ranges, bytes).map_err(|e| {
                            DataFusionError::from(e)
                                .context(format!("Pushing Parquet split ranges ({ctx})"))
                        })?;
                        meta.split_context.metrics.record_slice(meta.selected_rows);
                        if let Some(credit) = meta.credit_entry_bytes {
                            self.byte_progress.credit(credit);
                        }
                    } else {
                        decoder.push_ranges(ranges, bytes)?;
                        self.byte_progress.credit(entry.bytes);
                    }
                    let transport = match &self.direct_fetcher {
                        Some(fetcher) => Phase2Transport::Direct(Arc::clone(fetcher)),
                        None => {
                            let (tx, rx) = mpsc::unbounded_channel();
                            self.handback_rx = Some(rx);
                            Phase2Transport::Handback(tx)
                        }
                    };
                    let (row_context, decoded_lease, split_phase2, in_flight_guard) =
                        match split_meta {
                            Some(meta) => {
                                // Track filter planner BEFORE releasing active_split's
                                // FrontierGuard on the last slice so waiters never see
                                // zero active lanes + zero in-flight planners while
                                // this slice's buffers are still live.
                                let guard =
                                    meta.split_context.budget.track_filter_planner();
                                self.finish_published_split_slice(
                                    meta.is_last_slice,
                                    entry.fully_matched,
                                );
                                (
                                    Some((meta.rg_index, meta.row_range)),
                                    Some(meta.decoded_lease),
                                    Some(SplitPhase2State {
                                        broker: meta.broker,
                                        tracker: meta.tracker,
                                        phase2_lease: meta.phase2_lease,
                                        split_context: meta.split_context,
                                    }),
                                    Some(guard),
                                )
                            }
                            None => (None, None, None, None),
                        };
                    let rg_planner: Box<dyn MorselPlanner> =
                        Box::new(FilteredRowGroupPlanner {
                            decoder,
                            stage: RgStage::RunFilter,
                            pending_predicates: self.row_filter_context.num_predicates(),
                            buffered_bytes,
                            fetched: None,
                            transport,
                            projection: Arc::clone(&self.projection),
                            file_name: Arc::clone(&self.file_name),
                            row_context,
                            elapsed_compute: elapsed_compute.clone(),
                            cache_metrics: self.cache_metrics(),
                            decoded_lease,
                            split_phase2,
                            rg_handle,
                            _in_flight_guard: in_flight_guard,
                        });
                    let file_continuation: Box<dyn MorselPlanner> =
                        if self.has_more_work() {
                            Box::new(BufferedReaders::Filtered(self))
                        } else {
                            Box::new(EmptyDonePlanner)
                        };
                    // Two ready planners signal `[file_continuation, worker_filter_step]`
                    // to the native Parquet driver without touching stock `FileStream`.
                    return Ok(Some(
                        MorselPlan::new()
                            .with_planners(vec![file_continuation, rg_planner]),
                    ));
                }
                InFlight::FullyMatched {
                    mut decoder,
                    entry,
                    ranges,
                    bytes,
                    mut fetched_bytes,
                    split_meta,
                    use_bulk_projection,
                    mut rg_handle,
                } => {
                    fetched_bytes += bytes.iter().map(Bytes::len).sum::<usize>();
                    if let Some(meta) = &split_meta {
                        let ctx = format!(
                            "file={}, row_group={}, rows={}..{}",
                            self.file_name,
                            meta.rg_index,
                            meta.row_range.start,
                            meta.row_range.end
                        );
                        decoder.push_ranges(ranges, bytes).map_err(|e| {
                            DataFusionError::from(e)
                                .context(format!("Pushing Parquet split ranges ({ctx})"))
                        })?;
                    } else {
                        decoder.push_ranges(ranges, bytes)?;
                    }
                    match decoder.try_next_reader()? {
                        DecodeResult::NeedsData(more_ranges) => {
                            return Ok(Some(MorselPlan::new().with_pending_planner(
                                async move {
                                    let bytes = match split_meta {
                                        Some(mut meta) => {
                                            let ctx = format!(
                                                "file={}, row_group={}, rows={}..{}",
                                                self.file_name,
                                                meta.rg_index,
                                                meta.row_range.start,
                                                meta.row_range.end
                                            );
                                            let broker = Arc::clone(&meta.broker);
                                            let reader = &mut self.reader;
                                            let res = resolve_with_reader(
                                                &broker,
                                                &more_ranges,
                                                &mut meta.tracker,
                                                RangeAdmission::Phase1FollowUp,
                                                &ctx,
                                                reader,
                                            )
                                            .await?;
                                            self.in_flight =
                                                Some(InFlight::FullyMatched {
                                                    decoder,
                                                    entry,
                                                    ranges: more_ranges,
                                                    bytes: res,
                                                    fetched_bytes,
                                                    split_meta: Some(meta),
                                                    use_bulk_projection,
                                                    rg_handle,
                                                });
                                            return Ok(
                                                Box::new(BufferedReaders::Filtered(self))
                                                    as Box<dyn MorselPlanner>,
                                            );
                                        }
                                        None => {
                                            self.reader
                                                .get_byte_ranges(more_ranges.clone())
                                                .await?
                                        }
                                    };
                                    self.in_flight = Some(InFlight::FullyMatched {
                                        decoder,
                                        entry,
                                        ranges: more_ranges,
                                        bytes,
                                        fetched_bytes,
                                        split_meta: None,
                                        use_bulk_projection,
                                        rg_handle,
                                    });
                                    Ok(Box::new(BufferedReaders::Filtered(self))
                                        as Box<dyn MorselPlanner>)
                                },
                            )));
                        }
                        DecodeResult::Data(reader) => {
                            let (row_context, decoded_lease, in_flight_guard) =
                                match split_meta {
                                    Some(meta) => {
                                        meta.split_context
                                            .metrics
                                            .record_slice(meta.selected_rows);
                                        if let Some(credit) = meta.credit_entry_bytes {
                                            self.byte_progress.credit(credit);
                                        }
                                        // Register decode reader BEFORE dropping
                                        // active_split's FrontierGuard on last slice.
                                        let guard = meta
                                            .split_context
                                            .budget
                                            .track_decode_reader();
                                        self.finish_published_split_slice(
                                            meta.is_last_slice,
                                            entry.fully_matched,
                                        );
                                        drop(meta.phase2_lease);
                                        (
                                            Some((meta.rg_index, meta.row_range)),
                                            Some(meta.decoded_lease),
                                            Some(guard),
                                        )
                                    }
                                    None => {
                                        self.byte_progress.credit(entry.bytes);
                                        (None, None, None)
                                    }
                                };
                            if let Some(handle) = rg_handle.as_mut() {
                                handle.add_bytes_fetched(fetched_bytes as u64);
                            }
                            let projection = if use_bulk_projection {
                                self.bulk_projection.as_ref().map_or_else(
                                    || Arc::clone(&self.projection),
                                    Arc::clone,
                                )
                            } else {
                                Arc::clone(&self.projection)
                            };
                            let morsel = Box::new(BufferedReader {
                                reader,
                                compressed_bytes: fetched_bytes as u64,
                                projection,
                                file_name: Arc::clone(&self.file_name),
                                row_context,
                                elapsed_compute: elapsed_compute.clone(),
                                cache_metrics: Some(self.cache_metrics()),
                                decoded_lease,
                                in_flight_guard,
                                rg_handle,
                            });
                            return Ok(Some(self.publish_morsel(morsel)));
                        }
                        DecodeResult::Finished => {
                            if let Some(mut handle) = rg_handle.take() {
                                handle.add_bytes_fetched(fetched_bytes as u64);
                                handle.finish(0);
                            }
                            if let Some(meta) = split_meta {
                                if let Some(credit) = meta.credit_entry_bytes {
                                    self.byte_progress.credit(credit);
                                }
                                if meta.is_last_slice {
                                    self.active_split = None;
                                }
                            } else {
                                self.byte_progress.credit(entry.bytes);
                            }
                        }
                    }
                }
            }
        }

        loop {
            if let Some(mut split_rg) = self.active_split.take() {
                if split_rg.frontier_guard.is_none() {
                    let frontier_bytes = split_rg.plan.frontier_bytes;
                    let pool_frontier_bytes = split_rg.plan.pool_frontier_bytes;
                    let is_first_slice = split_rg.slices_published == 0;
                    let budget = Arc::clone(&split_rg.split_context.budget);
                    self.active_split = Some(split_rg);
                    return Ok(Some(MorselPlan::new().with_pending_planner(
                        async move {
                            match budget
                                .acquire_frontier(frontier_bytes, pool_frontier_bytes)
                                .await
                            {
                                Ok(guard) => {
                                    if let Some(rg) = self.active_split.as_mut() {
                                        rg.frontier_guard = Some(guard);
                                    }
                                }
                                Err(DataFusionError::ResourcesExhausted(_))
                                    if is_first_slice =>
                                {
                                    self.fallback_active_split_to_unsplit();
                                }
                                Err(err) => return Err(err),
                            }
                            Ok(Box::new(BufferedReaders::Filtered(self))
                                as Box<dyn MorselPlanner>)
                        },
                    )));
                }
                let Some(slice) = split_rg.plan.slices.pop_front() else {
                    self.byte_progress.credit(split_rg.entry.bytes);
                    continue;
                };
                let is_last = split_rg.plan.slices.is_empty();
                let is_first_slice = split_rg.slices_published == 0;
                let credit_entry_bytes = is_last.then_some(split_rg.entry.bytes);
                let install_filter = !split_rg.entry.fully_matched;
                let shared_keys = split_rg
                    .plan
                    .shared_ranges_per_slice
                    .get(slice.slice_idx)
                    .cloned()
                    .unwrap_or_default();
                let mut tracker =
                    Phase2SliceTracker::new(Arc::clone(&split_rg.broker), shared_keys);
                let ctx = format!(
                    "file={}, row_group={}, rows={}..{}",
                    self.file_name,
                    split_rg.rg_index,
                    slice.row_range.start,
                    slice.row_range.end
                );
                let mut no_handle = None;
                let mut decoder = self
                    .build_rg_decoder(
                        RowGroupSelection::new(split_rg.rg_index, Some(slice.selection)),
                        &self.projection,
                        install_filter,
                        &mut no_handle,
                    )
                    .map_err(|e| {
                        e.context(format!("Building Parquet split decoder ({ctx})"))
                    })?;
                let entry = split_rg.entry.clone();
                let rg_index = split_rg.rg_index;
                let decoded_estimate = slice.decoded_estimate;
                let all_ranges = slice.all_ranges;
                let broker = Arc::clone(&split_rg.broker);
                let split_context = Arc::clone(&split_rg.split_context);
                // Always retain `active_split` (including its `FrontierGuard`)
                // across the last slice's Phase-1 fetch until it is published!
                self.active_split = Some(split_rg);

                match decoder.try_next_reader().map_err(|e| {
                    DataFusionError::from(e)
                        .context(format!("Preparing Parquet split reader ({ctx})"))
                })? {
                    DecodeResult::NeedsData(ranges) => {
                        let phase1_set: HashSet<(u64, u64)> =
                            ranges.iter().map(|r| (r.start, r.end)).collect();
                        let phase2_candidates: Vec<(u64, u64)> = if install_filter {
                            all_ranges
                                .into_iter()
                                .filter(|k| !phase1_set.contains(k))
                                .collect()
                        } else {
                            Vec::new()
                        };
                        return Ok(Some(MorselPlan::new().with_pending_planner(
                            async move {
                                let mut admitted: Option<(
                                    Phase2BudgetLease,
                                    SplitDecodedLease,
                                )> = None;
                                let bytes = match resolve_with_reader(
                                    &broker,
                                    &ranges,
                                    &mut tracker,
                                    RangeAdmission::Phase1 {
                                        phase2_candidates: &phase2_candidates,
                                        decoded_estimate,
                                        out_leases: &mut admitted,
                                    },
                                    &ctx,
                                    &mut self.reader,
                                )
                                .await
                                {
                                    Ok(bytes) => bytes,
                                    Err(DataFusionError::ResourcesExhausted(_))
                                        if is_first_slice =>
                                    {
                                        drop(tracker);
                                        drop(admitted);
                                        self.fallback_active_split_to_unsplit();
                                        return Ok(Box::new(BufferedReaders::Filtered(
                                            self,
                                        ))
                                            as Box<dyn MorselPlanner>);
                                    }
                                    Err(err) => return Err(err),
                                };
                                let (phase2_lease, decoded_lease) = admitted
                                    .expect("Phase1 admission always populates leases");
                                let meta = Box::new(SplitSliceMeta {
                                    rg_index,
                                    row_range: slice.row_range,
                                    selected_rows: slice.selected_rows,
                                    is_last_slice: is_last,
                                    credit_entry_bytes,
                                    phase2_lease,
                                    decoded_lease,
                                    tracker,
                                    broker,
                                    split_context,
                                });
                                self.in_flight = Some(if install_filter {
                                    InFlight::Filtered {
                                        decoder,
                                        entry,
                                        ranges,
                                        bytes,
                                        split_meta: Some(meta),
                                        rg_handle: None,
                                    }
                                } else {
                                    InFlight::FullyMatched {
                                        decoder,
                                        entry,
                                        ranges,
                                        bytes,
                                        fetched_bytes: 0,
                                        split_meta: Some(meta),
                                        use_bulk_projection: false,
                                        rg_handle: None,
                                    }
                                });
                                Ok(Box::new(BufferedReaders::Filtered(self))
                                    as Box<dyn MorselPlanner>)
                            },
                        )));
                    }
                    DecodeResult::Data(reader) => {
                        let decoded_lease = split_context
                            .budget
                            .try_reserve_decoded_sync(decoded_estimate);
                        let Some(decoded_lease) = decoded_lease else {
                            let row_range = slice.row_range;
                            let selected_rows = slice.selected_rows;
                            let projection = Arc::clone(&self.projection);
                            let file_name = Arc::clone(&self.file_name);
                            let cache_metrics = self.cache_metrics();
                            let elapsed_compute_owned = elapsed_compute.clone();
                            return Ok(Some(MorselPlan::new().with_pending_planner(
                                async move {
                                    let decoded_lease = match split_context
                                        .budget
                                        .reserve_decoded(decoded_estimate)
                                        .await
                                    {
                                        Ok(lease) => lease,
                                        Err(DataFusionError::ResourcesExhausted(_))
                                            if is_first_slice =>
                                        {
                                            drop(tracker);
                                            self.fallback_active_split_to_unsplit();
                                            return Ok(
                                                Box::new(BufferedReaders::Filtered(self))
                                                    as Box<dyn MorselPlanner>,
                                            );
                                        }
                                        Err(err) => return Err(err),
                                    };
                                    split_context.metrics.record_slice(selected_rows);
                                    if let Some(credit) = credit_entry_bytes {
                                        self.byte_progress.credit(credit);
                                    }
                                    let in_flight_guard =
                                        split_context.budget.track_decode_reader();
                                    self.finish_published_split_slice(
                                        is_last,
                                        entry.fully_matched,
                                    );
                                    let morsel = Box::new(BufferedReader {
                                        reader,
                                        compressed_bytes: 0,
                                        projection,
                                        file_name,
                                        row_context: Some((rg_index, row_range)),
                                        elapsed_compute: elapsed_compute_owned,
                                        cache_metrics: Some(cache_metrics),
                                        decoded_lease: Some(decoded_lease),
                                        in_flight_guard: Some(in_flight_guard),
                                        rg_handle: None,
                                    });
                                    Ok(Box::new(PublishReadyMorsel {
                                        morsel: Some(morsel),
                                        continuation: self.has_more_work().then(|| {
                                            Box::new(BufferedReaders::Filtered(self))
                                                as Box<dyn MorselPlanner>
                                        }),
                                    })
                                        as Box<dyn MorselPlanner>)
                                },
                            )));
                        };
                        split_context.metrics.record_slice(slice.selected_rows);
                        if let Some(credit) = credit_entry_bytes {
                            self.byte_progress.credit(credit);
                        }
                        let in_flight_guard = split_context.budget.track_decode_reader();
                        self.finish_published_split_slice(is_last, entry.fully_matched);
                        let morsel = Box::new(BufferedReader {
                            reader,
                            compressed_bytes: 0,
                            projection: Arc::clone(&self.projection),
                            file_name: Arc::clone(&self.file_name),
                            row_context: Some((rg_index, slice.row_range)),
                            elapsed_compute: elapsed_compute.clone(),
                            cache_metrics: Some(self.cache_metrics()),
                            decoded_lease: Some(decoded_lease),
                            in_flight_guard: Some(in_flight_guard),
                            rg_handle: None,
                        });
                        return Ok(Some(self.publish_morsel(morsel)));
                    }
                    DecodeResult::Finished => {
                        if let Some(credit) = credit_entry_bytes {
                            self.byte_progress.credit(credit);
                        }
                        if is_last {
                            self.active_split = None;
                        }
                        continue;
                    }
                }
            }

            let Some(next) = self.pending.pop_front() else {
                return Ok(None);
            };
            match next {
                FilteredRgWork::Split {
                    rg_index,
                    original_selection,
                    entry,
                    plan,
                } => {
                    let split_context = Arc::clone(
                        self.split_context
                            .as_ref()
                            .expect("split work requires split_context"),
                    );
                    let frontier_guard = split_context.budget.try_acquire_frontier(
                        plan.frontier_bytes,
                        plan.pool_frontier_bytes,
                    );
                    let broker = Arc::new(AsyncRangeBroker::new(
                        plan.shared_consumer_counts.clone(),
                        Arc::clone(&split_context),
                    ));
                    self.active_split = Some(Box::new(ActiveSplitFilteredRg {
                        rg_index,
                        original_selection,
                        entry,
                        plan,
                        slices_published: 0,
                        frontier_guard,
                        broker,
                        split_context,
                    }));
                }
                FilteredRgWork::Unsplit { selection, entry } => {
                    let (install_filter, use_bulk_projection, mut rg_handle) =
                        if entry.fully_matched {
                            self.row_filter_skipped_fully_matched.add_one();
                            (false, false, None)
                        } else if let Some(adaptive) = &self.adaptive_context {
                            let rg_rows = self
                                .reader_metadata
                                .metadata()
                                .row_group(entry.rg_index)
                                .num_rows()
                                .max(0) as u64;
                            let can_skip = self.can_row_filter_skip_columns();
                            let handle = adaptive.assign_rg(
                                can_skip,
                                self.file_surviving_rgs,
                                rg_rows,
                            );
                            match handle.mode() {
                                RgFilterMode::RowFilter => (true, false, Some(handle)),
                                RgFilterMode::BulkDecode => (false, true, Some(handle)),
                            }
                        } else {
                            (true, false, None)
                        };
                    let decoder_projection = if use_bulk_projection {
                        self.bulk_projection.as_ref().unwrap_or(&self.projection)
                    } else {
                        &self.projection
                    };
                    let mut decoder = self.build_rg_decoder(
                        selection,
                        decoder_projection,
                        install_filter,
                        &mut rg_handle,
                    )?;
                    match decoder.try_next_reader()? {
                        DecodeResult::NeedsData(ranges) => {
                            return Ok(Some(MorselPlan::new().with_pending_planner(
                                async move {
                                    let bytes = self
                                        .reader
                                        .get_byte_ranges(ranges.clone())
                                        .await?;
                                    self.in_flight = Some(if install_filter {
                                        InFlight::Filtered {
                                            decoder,
                                            entry,
                                            ranges,
                                            bytes,
                                            split_meta: None,
                                            rg_handle,
                                        }
                                    } else {
                                        InFlight::FullyMatched {
                                            decoder,
                                            entry,
                                            ranges,
                                            bytes,
                                            fetched_bytes: 0,
                                            split_meta: None,
                                            use_bulk_projection,
                                            rg_handle,
                                        }
                                    });
                                    Ok(Box::new(BufferedReaders::Filtered(self))
                                        as Box<dyn MorselPlanner>)
                                },
                            )));
                        }
                        DecodeResult::Data(reader) => {
                            self.byte_progress.credit(entry.bytes);
                            let projection = if use_bulk_projection {
                                self.bulk_projection.as_ref().map_or_else(
                                    || Arc::clone(&self.projection),
                                    Arc::clone,
                                )
                            } else {
                                Arc::clone(&self.projection)
                            };
                            let morsel = Box::new(BufferedReader {
                                reader,
                                compressed_bytes: 0,
                                projection,
                                file_name: Arc::clone(&self.file_name),
                                row_context: None,
                                elapsed_compute: elapsed_compute.clone(),
                                cache_metrics: Some(self.cache_metrics()),
                                decoded_lease: None,
                                in_flight_guard: None,
                                rg_handle,
                            });
                            return Ok(Some(self.publish_morsel(morsel)));
                        }
                        DecodeResult::Finished => {
                            if let Some(handle) = rg_handle {
                                handle.finish(0);
                            }
                            self.byte_progress.credit(entry.bytes);
                        }
                    }
                }
            }
        }
    }
}

/// Helper to run `AsyncRangeBroker::resolve_ranges` against an exclusively
/// owned `Box<dyn AsyncFileReader>` on the file I/O lane.
async fn resolve_with_reader(
    broker: &Arc<AsyncRangeBroker>,
    requested: &[Range<u64>],
    tracker: &mut Phase2SliceTracker,
    admission: RangeAdmission<'_>,
    err_ctx: &str,
    reader: &mut Box<dyn AsyncFileReader>,
) -> Result<Vec<Bytes>> {
    // Since `AsyncFileReader::get_byte_ranges` takes `&mut self` (called at
    // most once inside `resolve_ranges` when there are misses, and possibly
    // once on takeover), wrap in a RefCell-free tokio::sync::Mutex that is
    // uncontended on this single lane.
    let reader = tokio::sync::Mutex::new(reader);
    broker
        .resolve_ranges(requested, tracker, admission, err_ctx, |ranges| {
            let reader = &reader;
            async move {
                let mut guard = reader.lock().await;
                guard
                    .get_byte_ranges(ranges)
                    .await
                    .map_err(DataFusionError::from)
            }
        })
        .await
}

enum RgStage {
    /// Ready to run `RowFilter` CPU evaluation synchronously on a worker.
    RunFilter,
    /// Returned from a Phase-2 `PendingMorselPlanner` on an I/O lane with newly
    /// fetched bytes in `fetched`.
    AfterIo,
}

struct SplitPhase2State {
    broker: Arc<AsyncRangeBroker>,
    tracker: Phase2SliceTracker,
    phase2_lease: Phase2BudgetLease,
    split_context: Arc<SplitContext>,
}

/// Per-row-group (or per-slice) state machine separating worker `RowFilter`
/// CPU work from I/O-lane Phase-2 range fetches.
///
/// Field drop order is load-bearing: `decoder`, `fetched`, `decoded_lease`, and
/// `split_phase2` (all holding buffer/pool reservations) MUST drop before
/// `_in_flight_guard` signals waiters.
struct FilteredRowGroupPlanner {
    decoder: ParquetPushDecoder,
    stage: RgStage,
    pending_predicates: usize,
    buffered_bytes: usize,
    fetched: Option<(Vec<Range<u64>>, Vec<Bytes>)>,
    transport: Phase2Transport,
    projection: Arc<DecoderProjection>,
    file_name: Arc<str>,
    row_context: Option<(usize, Range<usize>)>,
    elapsed_compute: Time,
    cache_metrics: PredicateCacheMetrics,
    decoded_lease: Option<SplitDecodedLease>,
    split_phase2: Option<SplitPhase2State>,
    rg_handle: Option<RgProbeHandle>,
    _in_flight_guard: Option<InFlightReaderGuard>,
}

impl fmt::Debug for FilteredRowGroupPlanner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FilteredRowGroupPlanner")
            .finish_non_exhaustive()
    }
}

impl FilteredRowGroupPlanner {
    fn err_context_str(&self) -> String {
        match &self.row_context {
            Some((rg, rows)) => format!(
                "Decoding Parquet split morsel from {} (row_group={rg}, rows={}..{})",
                self.file_name, rows.start, rows.end
            ),
            None => format!("Decoding Parquet morsel from {}", self.file_name),
        }
    }

    fn context_err(&self, err: DataFusionError) -> DataFusionError {
        err.context(self.err_context_str())
    }

    fn into_decode_morsel(self, reader: ParquetRecordBatchReader) -> Box<dyn Morsel> {
        let Self {
            decoder,
            buffered_bytes,
            fetched,
            projection,
            file_name,
            row_context,
            elapsed_compute,
            cache_metrics,
            decoded_lease,
            split_phase2,
            rg_handle,
            _in_flight_guard: filter_guard,
            ..
        } = self;
        let in_flight_guard = split_phase2
            .as_ref()
            .map(|s| s.split_context.budget.track_decode_reader());
        // Release decoder, fetched buffers, and any unused Phase-2 headroom
        // BEFORE dropping the filter-planner guard.
        drop(decoder);
        drop(fetched);
        drop(split_phase2);
        drop(filter_guard);
        Box::new(BufferedReader {
            reader,
            compressed_bytes: buffered_bytes as u64,
            projection,
            file_name,
            row_context,
            elapsed_compute,
            cache_metrics: Some(cache_metrics),
            decoded_lease,
            in_flight_guard,
            rg_handle,
        })
    }

    fn schedule_fetch(mut self: Box<Self>, ranges: Vec<Range<u64>>) -> MorselPlan {
        self.stage = RgStage::AfterIo;
        let transport = self.transport.clone();
        let err_ctx = self.err_context_str();
        MorselPlan::new().with_pending_planner(async move {
            let bytes = if let Some(split_phase2) = self.split_phase2.as_mut() {
                let broker = Arc::clone(&split_phase2.broker);
                broker
                    .resolve_ranges(
                        &ranges,
                        &mut split_phase2.tracker,
                        RangeAdmission::Phase2 {
                            phase2_lease: Some(&mut split_phase2.phase2_lease),
                        },
                        &err_ctx,
                        |fetch_ranges| {
                            let transport = transport.clone();
                            async move { transport.get_byte_ranges(fetch_ranges).await }
                        },
                    )
                    .await?
            } else {
                transport.get_byte_ranges(ranges.clone()).await?
            };
            self.fetched = Some((ranges, bytes));
            Ok(self as Box<dyn MorselPlanner>)
        })
    }
}

impl MorselPlanner for FilteredRowGroupPlanner {
    fn buffered_bytes_estimate(&self) -> usize {
        self.buffered_bytes
    }

    fn plan(mut self: Box<Self>) -> Result<Option<MorselPlan>> {
        let elapsed_compute = self.elapsed_compute.clone();
        let _timer = elapsed_compute.timer();

        match self.stage {
            RgStage::RunFilter => {
                let start = self
                    .rg_handle
                    .as_ref()
                    .is_some_and(RgProbeHandle::needs_worker_timing)
                    .then(Instant::now);
                let result = self
                    .decoder
                    .try_next_reader()
                    .map_err(|err| self.context_err(DataFusionError::from(err)))?;
                if let (Some(t0), Some(handle)) = (start, self.rg_handle.as_mut()) {
                    handle.add_cpu_ns(t0.elapsed().as_nanos() as u64);
                }
                self.pending_predicates = self.pending_predicates.saturating_sub(1);
                match result {
                    DecodeResult::Finished => {
                        self.cache_metrics.copy_to_gauges();
                        if let Some(handle) = self.rg_handle.take() {
                            handle.finish(0);
                        }
                        Ok(None)
                    }
                    DecodeResult::Data(reader) => {
                        let morsel = self.into_decode_morsel(reader);
                        Ok(Some(MorselPlan::new().with_morsels(vec![morsel])))
                    }
                    DecodeResult::NeedsData(ranges) => {
                        Ok(Some(self.schedule_fetch(ranges)))
                    }
                }
            }
            RgStage::AfterIo => {
                if let Some((ranges, bytes)) = self.fetched.take() {
                    let added = bytes.iter().map(Bytes::len).sum::<usize>();
                    self.buffered_bytes += added;
                    if let Some(handle) = self.rg_handle.as_mut() {
                        handle.add_bytes_fetched(added as u64);
                    }
                    self.decoder
                        .push_ranges(ranges, bytes)
                        .map_err(|err| self.context_err(DataFusionError::from(err)))?;
                }
                if self.pending_predicates > 0 {
                    // More RowFilter conjuncts remain: send back to pipeline
                    // workers without decoding on the I/O lane.
                    self.stage = RgStage::RunFilter;
                    return Ok(Some(
                        MorselPlan::new()
                            .with_planners(vec![Box::new(EmptyDonePlanner), self]),
                    ));
                }
                // All predicates have run: build the decode-only reader from
                // the buffered chunks without decoding output rows here.
                let start = self
                    .rg_handle
                    .as_ref()
                    .is_some_and(RgProbeHandle::needs_worker_timing)
                    .then(Instant::now);
                let result = self
                    .decoder
                    .try_next_reader()
                    .map_err(|err| self.context_err(DataFusionError::from(err)))?;
                if let (Some(t0), Some(handle)) = (start, self.rg_handle.as_mut()) {
                    handle.add_cpu_ns(t0.elapsed().as_nanos() as u64);
                }
                match result {
                    DecodeResult::Finished => {
                        self.cache_metrics.copy_to_gauges();
                        if let Some(handle) = self.rg_handle.take() {
                            handle.finish(0);
                        }
                        Ok(None)
                    }
                    DecodeResult::Data(reader) => {
                        let morsel = self.into_decode_morsel(reader);
                        Ok(Some(MorselPlan::new().with_morsels(vec![morsel])))
                    }
                    DecodeResult::NeedsData(more_ranges) => {
                        Ok(Some(self.schedule_fetch(more_ranges)))
                    }
                }
            }
        }
    }
}

/// Decode-ready morsel. Field drop order is load-bearing: `reader` holds
/// `TrackedBytes` leases and MUST drop before `decoded_lease` and
/// `in_flight_guard` so waiters never observe zero in-flight readers while
/// compressed bytes are still counted in `SplitBudget::live_bytes`.
struct BufferedReader {
    reader: ParquetRecordBatchReader,
    decoded_lease: Option<SplitDecodedLease>,
    in_flight_guard: Option<InFlightReaderGuard>,
    rg_handle: Option<RgProbeHandle>,
    compressed_bytes: u64,
    projection: Arc<DecoderProjection>,
    file_name: Arc<str>,
    row_context: Option<(usize, Range<usize>)>,
    elapsed_compute: Time,
    cache_metrics: Option<PredicateCacheMetrics>,
}

/// Iterator wrapper guaranteeing `reader` drops BEFORE `decoded_lease` and
/// `in_flight_guard` (struct fields drop strictly in declaration order).
struct BufferedReaderIter {
    reader: ParquetRecordBatchReader,
    _decoded_lease: Option<SplitDecodedLease>,
    _in_flight_guard: Option<InFlightReaderGuard>,
    projection: Arc<DecoderProjection>,
    file_name: Arc<str>,
    row_context: Option<(usize, Range<usize>)>,
    elapsed_compute: Time,
    cache_metrics: Option<PredicateCacheMetrics>,
}

impl Iterator for BufferedReaderIter {
    type Item = Result<RecordBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        let _timer = self.elapsed_compute.timer();
        let next = self.reader.next();
        if let Some(metrics) = &self.cache_metrics {
            metrics.copy_to_gauges();
        }
        next.map(|batch| {
            batch
                .map_err(DataFusionError::from)
                .and_then(|batch| self.projection.map(&batch))
                .map_err(|err| match &self.row_context {
                    Some((rg, rows)) => err.context(format!(
                        "Decoding Parquet split morsel from {} (row_group={rg}, rows={}..{})",
                        self.file_name, rows.start, rows.end
                    )),
                    None => err.context(format!(
                        "Decoding Parquet morsel from {}",
                        self.file_name
                    )),
                })
        })
    }
}

impl fmt::Debug for BufferedReader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BufferedReader").finish_non_exhaustive()
    }
}

impl Morsel for BufferedReader {
    fn buffered_bytes_estimate(&self) -> usize {
        self.compressed_bytes as usize
    }

    fn into_stream(self: Box<Self>) -> stream::BoxStream<'static, Result<RecordBatch>> {
        let Self {
            reader,
            decoded_lease,
            in_flight_guard,
            rg_handle,
            projection,
            file_name,
            row_context,
            elapsed_compute,
            cache_metrics,
            ..
        } = *self;
        if let Some(handle) = rg_handle {
            publish_into_stream_handle(handle);
        }
        stream::iter(BufferedReaderIter {
            reader,
            _decoded_lease: decoded_lease,
            _in_flight_guard: in_flight_guard,
            projection,
            file_name,
            row_context,
            elapsed_compute,
            cache_metrics,
        })
        .boxed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use arrow::array::{
        Array, Int32Array, Int64Array, ListArray, StringArray, StructArray,
    };
    use arrow::buffer::{BooleanBuffer, OffsetBuffer};
    use arrow::datatypes::{DataType, Field, Fields, Schema};
    use datafusion_common_runtime::SpawnedTask;
    use datafusion_execution::memory_pool::GreedyMemoryPool;
    use datafusion_physical_expr::projection::ProjectionExprs;
    use futures::future::BoxFuture;
    use parquet::arrow::ArrowWriter;
    use parquet::arrow::arrow_reader::ArrowReaderOptions;
    use parquet::file::metadata::{PageIndexPolicy, ParquetMetaDataReader};
    use parquet::file::properties::{EnabledStatistics, WriterProperties};
    use tokio::sync::Barrier;

    fn write_parquet_bytes(batch: &RecordBatch, props: WriterProperties) -> Bytes {
        let mut buf = Vec::new();
        let mut writer =
            ArrowWriter::try_new(&mut buf, batch.schema(), Some(props)).unwrap();
        writer.write(batch).unwrap();
        writer.close().unwrap();
        Bytes::from(buf)
    }

    fn load_metadata(data: &Bytes, load_offset_index: bool) -> ArrowReaderMetadata {
        let mut reader = ParquetMetaDataReader::new();
        if load_offset_index {
            reader = reader
                .with_offset_index_policy(PageIndexPolicy::Optional)
                .with_column_index_policy(PageIndexPolicy::Optional);
        }
        reader.try_parse(data).unwrap();
        let meta = Arc::new(reader.finish().unwrap());
        let opts =
            ArrowReaderOptions::new().with_page_index_policy(if load_offset_index {
                PageIndexPolicy::Optional
            } else {
                PageIndexPolicy::Skip
            });
        ArrowReaderMetadata::try_new(meta, opts).unwrap()
    }

    #[derive(Clone)]
    struct RecordingReader {
        data: Bytes,
        recorded: Arc<Mutex<Vec<Range<u64>>>>,
        fail_on_call: Option<usize>,
        calls: Arc<AtomicUsize>,
    }

    impl RecordingReader {
        fn new(data: Bytes) -> Self {
            Self {
                data,
                recorded: Arc::new(Mutex::new(Vec::new())),
                fail_on_call: None,
                calls: Arc::new(AtomicUsize::new(0)),
            }
        }

        fn with_fail_on_call(mut self, call_idx: usize) -> Self {
            self.fail_on_call = Some(call_idx);
            self
        }

        fn recorded_ranges(&self) -> Vec<Range<u64>> {
            self.recorded.lock().clone()
        }
    }

    impl AsyncFileReader for RecordingReader {
        fn get_bytes(
            &mut self,
            range: Range<u64>,
        ) -> BoxFuture<'_, parquet::errors::Result<Bytes>> {
            let start = range.start as usize;
            let end = range.end as usize;
            let slice = self.data.slice(start..end);
            Box::pin(async move { Ok(slice) })
        }

        fn get_byte_ranges(
            &mut self,
            ranges: Vec<Range<u64>>,
        ) -> BoxFuture<'_, parquet::errors::Result<Vec<Bytes>>> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            if self.fail_on_call == Some(call) {
                return Box::pin(async move {
                    Err(ParquetError::External(Box::new(std::io::Error::other(
                        "injected storage read failure",
                    ))))
                });
            }
            self.recorded.lock().extend(ranges.iter().cloned());
            let out: Vec<Bytes> = ranges
                .into_iter()
                .map(|r| self.data.slice(r.start as usize..r.end as usize))
                .collect();
            Box::pin(async move { Ok(out) })
        }

        fn get_metadata<'a>(
            &'a mut self,
            _options: Option<&'a ArrowReaderOptions>,
        ) -> BoxFuture<'a, parquet::errors::Result<Arc<ParquetMetaData>>> {
            let meta = load_metadata(&self.data, true).metadata().clone();
            Box::pin(async move { Ok(meta) })
        }
    }

    async fn drive_planner_collect(
        mut planner: Box<dyn MorselPlanner>,
    ) -> Result<(Vec<RecordBatch>, usize)> {
        let mut batches = Vec::new();
        let mut morsel_count = 0usize;
        while let Some(mut step) = planner.plan()? {
            for morsel in step.take_morsels() {
                morsel_count += 1;
                let mut stream = morsel.into_stream();
                while let Some(batch) = stream.next().await {
                    batches.push(batch?);
                }
            }
            let mut ready = step.take_ready_planners();
            let pending = step.take_pending_planner();
            planner = if let Some(io) = pending {
                io.await?
            } else if let Some(next) = ready.pop() {
                next
            } else {
                break;
            };
        }
        Ok((batches, morsel_count))
    }

    fn selected_row_indices(selection: &RowSelection, total_rows: usize) -> Vec<usize> {
        let mut out = Vec::new();
        let mut pos = 0usize;
        for sel in selection.iter() {
            let end = (pos + sel.row_count).min(total_rows);
            if !sel.skip {
                out.extend(pos..end);
            }
            pos = end;
        }
        out
    }

    #[test]
    fn split_selection_disjoint_and_exact_union() {
        // Full RG with short final slice
        let slices = split_selection_into_slices(None, 105, 30);
        assert_eq!(slices.len(), 4);
        assert_eq!(slices[3].row_range, 90..105);
        assert_eq!(slices[3].selected_rows, 15);
        let mut covered = Vec::new();
        for s in &slices {
            let rows = selected_row_indices(&s.selection, 105);
            assert_eq!(rows.len(), s.selected_rows);
            covered.extend(rows);
        }
        assert_eq!(covered, (0..105).collect::<Vec<_>>());

        // Zero-row input
        assert!(split_selection_into_slices(None, 0, 30).is_empty());

        // Sparse selector-backed selection with completely pruned middle slice
        let sparse = RowSelection::from(vec![
            RowSelector::skip(5),
            RowSelector::select(10), // 5..15 (in slice 0: 0..30)
            RowSelector::skip(55),   // 15..70 (slice 1: 30..60 is empty!)
            RowSelector::select(25), // 70..95 (straddles 60..90 and 90..105)
            RowSelector::skip(10),   // 95..105
        ]);
        let slices = split_selection_into_slices(Some(&sparse), 105, 30);
        assert_eq!(slices.len(), 3);
        assert_eq!(slices[0].row_range, 0..30);
        assert_eq!(slices[1].row_range, 60..90);
        assert_eq!(slices[2].row_range, 90..105);
        let expected = selected_row_indices(&sparse, 105);
        let mut actual = Vec::new();
        for s in &slices {
            assert_eq!(s.selection.total_row_count(), 105);
            actual.extend(selected_row_indices(&s.selection, 105));
        }
        assert_eq!(actual, expected);

        // Mask-backed selection
        let bits: Vec<bool> = (0..100).map(|i| i % 7 == 0 || i == 99).collect();
        let mask_sel = RowSelection::from(BooleanBuffer::from(bits));
        let slices = split_selection_into_slices(Some(&mask_sel), 100, 25);
        let mut actual = Vec::new();
        for s in &slices {
            assert_eq!(s.selection.total_row_count(), 100);
            actual.extend(selected_row_indices(&s.selection, 100));
        }
        assert_eq!(actual, selected_row_indices(&mask_sel, 100));

        // Short selection (fewer rows than total_rows -> implicit trailing skip)
        let short = RowSelection::from(vec![RowSelector::select(12)]);
        let slices = split_selection_into_slices(Some(&short), 100, 25);
        assert_eq!(slices.len(), 1);
        assert_eq!(
            selected_row_indices(&slices[0].selection, 100),
            (0..12).collect::<Vec<_>>()
        );
    }

    #[test]
    fn choose_aligned_boundaries_prefers_tight_window_and_widens_for_large_pages() {
        let loc = |offset, size, first_row| PageLocation {
            offset,
            compressed_page_size: size,
            first_row_index: first_row,
        };
        // Wide column has pages every 140 rows (size 1000); narrow column has pages
        // every 275 rows (size 400). Target=128 -> tight window [64, 192] snaps to
        // wide column's 140-row page boundaries.
        let wide: Vec<PageLocation> =
            (0..10).map(|i| loc(i * 1000, 1000, i * 140)).collect();
        let narrow: Vec<PageLocation> =
            (0..6).map(|i| loc(i * 400, 400, i * 275)).collect();
        let ranges = choose_aligned_boundaries(1400, 128, &[&wide, &narrow]);
        assert!(ranges.len() >= 8);
        assert_eq!(ranges[0], 0..140);
        assert_eq!(ranges.last().unwrap().end, 1400);
        for w in ranges.windows(2) {
            assert_eq!(w[0].end, w[1].start);
        }

        // Narrow-only query (275-row pages): tight window [64, 192] has no boundary,
        // so it widens to [64, 384] and snaps cleanly to 275-row page boundaries.
        let narrow_ranges = choose_aligned_boundaries(1400, 128, &[&narrow]);
        assert_eq!(narrow_ranges[0], 0..275);
        assert_eq!(narrow_ranges[1], 275..550);
        assert_eq!(narrow_ranges.last().unwrap().end, 1400);
    }

    #[test]
    fn validate_offset_index_edge_cases() {
        let loc = |offset, compressed_page_size, first_row_index| PageLocation {
            offset,
            compressed_page_size,
            first_row_index,
        };
        let valid = vec![loc(100, 50, 0), loc(150, 60, 10)];
        assert!(validate_offset_index(&valid, 20, 100, 110));
        // Dictionary prefix before first page is valid
        assert!(validate_offset_index(&valid, 20, 80, 130));
        // Empty locations
        assert!(!validate_offset_index(&[], 20, 100, 110));
        // First page does not start at row 0
        assert!(!validate_offset_index(&[loc(100, 50, 1)], 20, 100, 110));
        // Non-monotonic first_row_index
        assert!(!validate_offset_index(
            &[loc(100, 50, 0), loc(150, 50, 0)],
            20,
            100,
            110
        ));
        // Overlapping byte ranges
        assert!(!validate_offset_index(
            &[loc(100, 50, 0), loc(140, 50, 10)],
            20,
            100,
            110
        ));
        // Out of chunk bounds
        assert!(!validate_offset_index(&valid, 20, 100, 100));
        // Zero compressed_page_size
        assert!(!validate_offset_index(&[loc(100, 0, 0)], 20, 100, 110));
    }

    #[test]
    fn coalesce_misses_respects_boundaries_gaps_and_lifetimes() {
        let mut last_consumer = HashMap::new();
        // dict (0..100) retained through slice 5; interior pages (100..200, 200..300)
        // end at slice 0; gap at 300..350; interior (350..450); boundary page (450..550)
        // retained through slice 1.
        last_consumer.insert((0, 100), 5);
        last_consumer.insert((100, 200), 0);
        last_consumer.insert((200, 300), 0);
        last_consumer.insert((350, 450), 0);
        last_consumer.insert((450, 550), 1);

        let misses = vec![0..100, 100..200, 200..300, 350..450, 450..550];
        let groups = coalesce_misses(misses, &last_consumer, 0, 1024);
        let ranges: Vec<Range<u64>> = groups.iter().map(|g| g.range.clone()).collect();
        // dict kept separate from interior; interior 100..200 + 200..300 merged;
        // gap 300..350 NOT fetched; boundary 450..550 kept separate from interior 350..450.
        assert_eq!(ranges, vec![0..100, 100..300, 350..450, 450..550]);

        // Max coalesced size cap splits adjacent interior misses and allows a single oversized page
        let big = vec![0..600, 600..1200, 1200..3000];
        let mut lc = HashMap::new();
        lc.insert((0, 600), 0);
        lc.insert((600, 1200), 0);
        lc.insert((1200, 3000), 0);
        let groups = coalesce_misses(big, &lc, 0, 1000);
        let ranges: Vec<Range<u64>> = groups.iter().map(|g| g.range.clone()).collect();
        assert_eq!(ranges, vec![0..600, 600..1200, 1200..3000]);
    }

    #[test]
    fn plan_row_group_splits_eligibility_and_rejections() {
        let n = 512usize;
        let ids = Arc::new(Int64Array::from_iter_values(0..n as i64));
        let vals = Arc::new(Int32Array::from_iter_values((0..n as i32).map(|x| x * 3)));
        let strings = Arc::new(StringArray::from(vec!["hello"; n]));
        let struct_arr = Arc::new(StructArray::from(vec![(
            Arc::new(Field::new("s_id", DataType::Int32, false)),
            vals.clone() as _,
        )]));
        let list_offsets = OffsetBuffer::from_lengths(vec![1usize; n]);
        let list_arr = Arc::new(ListArray::new(
            Arc::new(Field::new("item", DataType::Int32, false)),
            list_offsets,
            vals.clone(),
            None,
        ));
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("val", DataType::Int32, false),
            Field::new("s", DataType::Utf8, false),
            Field::new(
                "st",
                DataType::Struct(Fields::from(vec![Field::new(
                    "s_id",
                    DataType::Int32,
                    false,
                )])),
                false,
            ),
            Field::new(
                "lst",
                DataType::List(Arc::new(Field::new("item", DataType::Int32, false))),
                false,
            ),
        ]));
        let batch =
            RecordBatch::try_new(schema, vec![ids, vals, strings, struct_arr, list_arr])
                .unwrap();

        let props = WriterProperties::builder()
            .set_max_row_group_row_count(Some(n))
            .set_write_batch_size(32)
            .set_data_page_row_count_limit(64)
            .set_data_page_size_limit(256)
            .set_statistics_enabled(EnabledStatistics::Page)
            .set_dictionary_enabled(false)
            .build();
        let data = write_parquet_bytes(&batch, props);
        let meta_indexed = load_metadata(&data, true);
        let meta_unindexed = load_metadata(&data, false);

        let cfg = ParquetSplitConfig {
            enabled: true,
            min_row_group_rows: 256,
            target_slice_rows: 128,
            ..Default::default()
        };

        // Eligible: flat fixed-width leaves 0 & 1
        assert!(
            plan_row_group_splits(
                meta_indexed.metadata(),
                0,
                None,
                |i| i == 0 || i == 1,
                |_| false,
                &cfg,
                true,
                64,
                None,
            )
            .is_ok()
        );

        // Disabled
        let mut disabled = cfg.clone();
        disabled.enabled = false;
        assert_eq!(
            plan_row_group_splits(
                meta_indexed.metadata(),
                0,
                None,
                |i| i == 0,
                |_| false,
                &disabled,
                true,
                64,
                None,
            )
            .err(),
            Some(UnsplitReason::Disabled)
        );

        // Page index disabled
        assert_eq!(
            plan_row_group_splits(
                meta_indexed.metadata(),
                0,
                None,
                |i| i == 0,
                |_| false,
                &cfg,
                false,
                64,
                None,
            )
            .err(),
            Some(UnsplitReason::PageIndexDisabled)
        );

        // Below threshold
        let mut high_thresh = cfg.clone();
        high_thresh.min_row_group_rows = 1024;
        assert_eq!(
            plan_row_group_splits(
                meta_indexed.metadata(),
                0,
                None,
                |i| i == 0,
                |_| false,
                &high_thresh,
                true,
                64,
                None,
            )
            .err(),
            Some(UnsplitReason::BelowThreshold)
        );

        // Metadata-only (0 leaves)
        assert_eq!(
            plan_row_group_splits(
                meta_indexed.metadata(),
                0,
                None,
                |_| false,
                |_| false,
                &cfg,
                true,
                64,
                None,
            )
            .err(),
            Some(UnsplitReason::MetadataOnly)
        );

        // Unsupported types: UTF-8 (leaf 2), Struct child (leaf 3), List child (leaf 4)
        for bad_leaf in [2usize, 3, 4] {
            assert_eq!(
                plan_row_group_splits(
                    meta_indexed.metadata(),
                    0,
                    None,
                    |i| i == bad_leaf,
                    |_| false,
                    &cfg,
                    true,
                    64,
                    None,
                )
                .err(),
                Some(UnsplitReason::UnsupportedColumnType),
                "leaf {bad_leaf}"
            );
        }

        // Missing offset index
        assert_eq!(
            plan_row_group_splits(
                meta_unindexed.metadata(),
                0,
                None,
                |i| i == 0,
                |_| false,
                &cfg,
                true,
                64,
                None,
            )
            .err(),
            Some(UnsplitReason::MissingOffsetIndex)
        );

        // Single surviving slice after selection pruning
        let single_slice_sel =
            RowSelection::from(vec![RowSelector::select(10), RowSelector::skip(n - 10)]);
        assert_eq!(
            plan_row_group_splits(
                meta_indexed.metadata(),
                0,
                Some(&single_slice_sel),
                |i| i == 0,
                |_| false,
                &cfg,
                true,
                64,
                None,
            )
            .err(),
            Some(UnsplitReason::SingleSlice)
        );

        // Decoded workspace cap
        let mut tiny_decode = cfg.clone();
        tiny_decode.max_decoded_bytes = 64;
        assert_eq!(
            plan_row_group_splits(
                meta_indexed.metadata(),
                0,
                None,
                |i| i == 0,
                |_| false,
                &tiny_decode,
                true,
                64,
                None,
            )
            .err(),
            Some(UnsplitReason::DecodedWorkspaceCap)
        );

        // Per-slice compressed cap
        let mut tiny_slice = cfg.clone();
        tiny_slice.max_slice_compressed_bytes = 1;
        assert_eq!(
            plan_row_group_splits(
                meta_indexed.metadata(),
                0,
                None,
                |i| i == 0,
                |_| false,
                &tiny_slice,
                true,
                64,
                None,
            )
            .err(),
            Some(UnsplitReason::SliceCompressedCap)
        );

        // Finite memory pool smaller than pool frontier rejected at planning time
        assert_eq!(
            plan_row_group_splits(
                meta_indexed.metadata(),
                0,
                None,
                |i| i == 0,
                |_| false,
                &cfg,
                true,
                64,
                Some(64),
            )
            .err(),
            Some(UnsplitReason::MemoryPoolCap)
        );

        // Single giant page rejected by decode amplification heuristic (> 2x)
        let one_page_props = WriterProperties::builder()
            .set_max_row_group_row_count(Some(n))
            .set_data_page_row_count_limit(n * 4)
            .set_data_page_size_limit(1 << 20)
            .set_statistics_enabled(EnabledStatistics::Page)
            .set_dictionary_enabled(false)
            .build();
        let one_page_data = write_parquet_bytes(&batch, one_page_props);
        let one_page_meta = load_metadata(&one_page_data, true);
        assert_eq!(
            plan_row_group_splits(
                one_page_meta.metadata(),
                0,
                None,
                |i| i == 0,
                |_| false,
                &cfg,
                true,
                64,
                None,
            )
            .err(),
            Some(UnsplitReason::DecodeAmplification)
        );
    }

    fn build_indexed_dict_fixture(n: usize) -> (Bytes, Arc<Schema>) {
        let ids = Arc::new(Int64Array::from_iter_values(0..n as i64));
        // Repeated small set of values -> dictionary encoded
        let dict_col = Arc::new(Int32Array::from(
            (0..n)
                .map(|i| {
                    if i % 13 == 0 {
                        None
                    } else {
                        Some((i % 7) as i32)
                    }
                })
                .collect::<Vec<_>>(),
        ));
        // Unprojected column to verify we never fetch its byte range
        let unused =
            Arc::new(Int64Array::from_iter_values((0..n as i64).map(|x| x * 99)));
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("dict_col", DataType::Int32, true),
            Field::new("unused", DataType::Int64, false),
        ]));
        let batch =
            RecordBatch::try_new(Arc::clone(&schema), vec![ids, dict_col, unused])
                .unwrap();
        // Force multiple small pages with boundaries misaligned relative to 400-row slices
        let props = WriterProperties::builder()
            .set_max_row_group_row_count(Some(n))
            .set_data_page_row_count_limit(70)
            .set_data_page_size_limit(256)
            .set_dictionary_page_size_limit(256)
            .set_write_batch_size(35)
            .set_statistics_enabled(EnabledStatistics::Page)
            .set_dictionary_enabled(true)
            .build();
        (write_parquet_bytes(&batch, props), schema)
    }

    fn make_split_plain_planner(
        data: &Bytes,
        reader: RecordingReader,
        proj_indices: &[usize],
        split_ctx: Arc<SplitContext>,
        metrics: &ExecutionPlanMetricsSet,
    ) -> Box<dyn MorselPlanner> {
        let reader_metadata = load_metadata(data, true);
        let file_schema = Arc::clone(reader_metadata.schema());
        let output_schema = Arc::new(file_schema.project(proj_indices).unwrap());
        let projection_exprs = ProjectionExprs::from_indices(proj_indices, &file_schema);
        let projection = Arc::new(
            DecoderProjection::try_new(
                &projection_exprs,
                &file_schema,
                reader_metadata.parquet_schema(),
                &output_schema,
                None,
            )
            .unwrap(),
        );
        let rg_bytes = reader_metadata.metadata().row_group(0).compressed_size() as u64;
        let pending = vec![(
            RowGroupSelection::new(0, None),
            RgPlanEntry {
                rg_index: 0,
                bytes: rg_bytes,
                fully_matched: false,
            },
        )];
        let cfg = SplitPlainBufferedReadersConfig {
            pending,
            reader_metadata,
            projection,
            batch_size: 64,
            force_filter_selections: false,
            enable_page_index: true,
            arrow_reader_metrics: ArrowReaderMetrics::enabled(),
            file_name: Arc::from("fixture.parquet"),
            baseline: BaselineMetrics::new(metrics, 0),
            byte_progress: ByteProgress::new(
                rg_bytes,
                MetricBuilder::new(metrics).counter("bytes_processed", 0),
            ),
            reader: Box::new(reader),
            split_context: split_ctx,
        };
        let readers = BufferedReaders::try_split_plain(cfg)
            .unwrap_or_else(|_| panic!("expected row group to split"));
        Box::new(readers)
    }

    #[tokio::test]
    async fn recording_reader_shares_boundary_page_and_dictionary_once() {
        let n = 2400usize;
        let (data, _schema) = build_indexed_dict_fixture(n);
        let meta = load_metadata(&data, true);
        // Confirm dict_col actually wrote a dictionary prefix and multiple pages
        {
            let rg_meta = meta.metadata().row_group(0);
            let (s, _) = rg_meta.column(1).byte_range();
            let locs = meta
                .metadata()
                .page_index()
                .unwrap()
                .offset_index(0, 1)
                .unwrap()
                .page_locations();
            assert!(
                dictionary_range(s, locs).is_some(),
                "expected dict_col to have a dictionary page"
            );
            assert!(locs.len() > 2, "expected multiple data pages for dict_col");
        }
        let unused_range = {
            let (s, l) = meta.metadata().row_group(0).column(2).byte_range();
            s..s + l
        };

        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(64 << 20));
        let metrics = ExecutionPlanMetricsSet::new();
        let cfg = ParquetSplitConfig {
            enabled: true,
            min_row_group_rows: 800,
            target_slice_rows: 400, // 6 slices; misaligned with 70-row pages!
            ..Default::default()
        };
        let split_ctx = Arc::new(SplitContext::new(
            cfg,
            Some(Arc::clone(&pool)),
            metrics.clone(),
        ));
        let recorder = RecordingReader::new(data.clone());
        let planner = make_split_plain_planner(
            &data,
            recorder.clone(),
            &[0, 1],
            Arc::clone(&split_ctx),
            &metrics,
        );

        let (batches, morsels) = drive_planner_collect(planner).await.unwrap();
        assert!(
            morsels >= 4,
            "expected multiple split slices, got {morsels}"
        );
        let mut ids = Vec::new();
        for b in &batches {
            let id_arr = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
            let dict_arr = b.column(1).as_any().downcast_ref::<Int32Array>().unwrap();
            for row in 0..b.num_rows() {
                let id = id_arr.value(row);
                ids.push(id);
                if id % 13 == 0 {
                    assert!(dict_arr.is_null(row));
                } else {
                    assert_eq!(dict_arr.value(row), (id % 7) as i32);
                }
            }
        }
        assert_eq!(ids, (0..n as i64).collect::<Vec<_>>());

        // Verify no byte was fetched twice (every recorded byte range is disjoint)
        // and the unprojected column was never read.
        let mut recorded = recorder.recorded_ranges();
        assert!(!recorded.is_empty());
        recorded.sort_unstable_by_key(|r| (r.start, r.end));
        for win in recorded.windows(2) {
            assert!(
                win[0].end <= win[1].start,
                "duplicate/overlapping storage read: {:?} and {:?}",
                win[0],
                win[1]
            );
        }
        for r in &recorded {
            assert!(
                r.end <= unused_range.start || r.start >= unused_range.end,
                "unprojected column range {unused_range:?} overlapped read {r:?}"
            );
        }

        // Verify metrics & cleanup
        let snap = metrics.clone_inner();
        assert_eq!(
            snap.sum_by_name("parquet_split_row_groups")
                .unwrap()
                .as_usize(),
            1
        );
        assert_eq!(
            snap.sum_by_name("parquet_split_slices").unwrap().as_usize(),
            morsels
        );
        assert_eq!(
            snap.sum_by_name("parquet_split_selected_rows")
                .unwrap()
                .as_usize(),
            n
        );
        assert!(
            snap.sum_by_name("parquet_split_reused_bytes")
                .unwrap()
                .as_usize()
                > 0,
            "expected shared boundary/dictionary reuse"
        );
        assert!(
            snap.sum_by_name("parquet_split_peak_unique_compressed_bytes")
                .unwrap()
                .as_usize()
                > 0
        );
        // Logical byte progress credited once for the RG.
        let rg_bytes = meta.metadata().row_group(0).compressed_size() as usize;
        assert_eq!(
            snap.sum_by_name("bytes_processed").unwrap().as_usize(),
            rg_bytes
        );
        assert_eq!(split_ctx.budget.live_bytes(), 0);
        assert_eq!(split_ctx.budget.admitted_frontier_bytes(), 0);
        assert_eq!(pool.reserved(), 0);
    }

    #[tokio::test]
    async fn tiny_buffer_cap_multiple_lanes_makes_progress_without_deadlock() {
        let n = 2400usize;
        let (data, _) = build_indexed_dict_fixture(n);
        let meta = load_metadata(&data, true);
        let probe_cfg = ParquetSplitConfig {
            enabled: true,
            min_row_group_rows: 800,
            target_slice_rows: 400,
            ..Default::default()
        };
        let probe_plan = plan_row_group_splits(
            meta.metadata(),
            0,
            None,
            |i| i == 0 || i == 1,
            |_| false,
            &probe_cfg,
            true,
            64,
            None,
        )
        .unwrap();
        // Set query-wide cap just large enough for ONE lane's frontier, forcing
        // 4 concurrent lanes to serialize frontier admission without deadlocking.
        let tight_cap = probe_plan.frontier_bytes + 64;
        let cfg = ParquetSplitConfig {
            split_buffer_cap_bytes: tight_cap,
            ..probe_cfg
        };
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(64 << 20));
        let metrics = ExecutionPlanMetricsSet::new();
        let split_ctx = Arc::new(SplitContext::new(
            cfg,
            Some(Arc::clone(&pool)),
            metrics.clone(),
        ));

        let mut handles = Vec::new();
        for _ in 0..4 {
            let planner = make_split_plain_planner(
                &data,
                RecordingReader::new(data.clone()),
                &[0, 1],
                Arc::clone(&split_ctx),
                &metrics,
            );
            handles.push(SpawnedTask::spawn(async move {
                drive_planner_collect(planner).await
            }));
        }

        for h in handles {
            let (batches, morsels) = tokio::time::timeout(Duration::from_secs(10), h)
                .await
                .expect("timed out: possible deadlock")
                .unwrap()
                .unwrap();
            assert!(
                morsels >= 4,
                "expected multiple split slices, got {morsels}"
            );
            let total_rows: usize = batches.iter().map(RecordBatch::num_rows).sum();
            assert_eq!(total_rows, n);
        }

        assert_eq!(split_ctx.budget.live_bytes(), 0);
        assert_eq!(split_ctx.budget.admitted_frontier_bytes(), 0);
        assert_eq!(pool.reserved(), 0);
    }

    #[tokio::test]
    async fn memory_pool_exhaustion_falls_back_on_slice0_and_errors_mid_rg() {
        let n = 2400usize;
        let (data, _) = build_indexed_dict_fixture(n);
        let cfg = ParquetSplitConfig {
            enabled: true,
            min_row_group_rows: 800,
            target_slice_rows: 400,
            ..Default::default()
        };

        // Case 1: Pool fits pool_frontier statically, then an external consumer
        // occupies the pool before slice 0 reserves -> cleanly falls back to
        // unsplit reader before publishing any slice, returning all rows.
        let pool_cap1 = 1 << 20;
        let tight_pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(pool_cap1));
        let metrics = ExecutionPlanMetricsSet::new();
        let split_ctx = Arc::new(SplitContext::new(
            cfg.clone(),
            Some(Arc::clone(&tight_pool)),
            metrics.clone(),
        ));
        let planner = make_split_plain_planner(
            &data,
            RecordingReader::new(data.clone()),
            &[0, 1],
            Arc::clone(&split_ctx),
            &metrics,
        );
        let external_blocker = MemoryConsumer::new("external").register(&tight_pool);
        external_blocker.try_grow(pool_cap1 - 1).unwrap();
        let (batches, morsels) =
            tokio::time::timeout(Duration::from_secs(5), drive_planner_collect(planner))
                .await
                .expect("must not deadlock on slice-0 pool exhaustion")
                .expect("slice-0 pool denial should fall back to unsplit reader");
        assert_eq!(morsels, 1);
        assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), n);
        let snap = metrics.clone_inner();
        assert_eq!(
            snap.sum_by_name("parquet_split_row_groups")
                .map_or(0, |v| v.as_usize()),
            0
        );
        assert_eq!(split_ctx.budget.live_bytes(), 0);
        assert_eq!(split_ctx.budget.admitted_frontier_bytes(), 0);
        drop(external_blocker);
        assert_eq!(tight_pool.reserved(), 0);

        // Case 2: Pool fits slice 0, then external consumer drains the pool
        // before slice 1 -> MUST return ResourcesExhausted without deadlocking
        // or replaying slice 0.
        let pool_cap = 1 << 20;
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(pool_cap));
        let metrics2 = ExecutionPlanMetricsSet::new();
        let split_ctx2 = Arc::new(SplitContext::new(
            cfg,
            Some(Arc::clone(&pool)),
            metrics2.clone(),
        ));
        let mut planner = make_split_plain_planner(
            &data,
            RecordingReader::new(data.clone()),
            &[0, 1],
            Arc::clone(&split_ctx2),
            &metrics2,
        );
        // Drain slice 0 completely so slices_published == 1 and in_flight == 0.
        loop {
            let mut step = planner.plan().unwrap().unwrap();
            let morsels = step.take_morsels();
            let mut ready = step.take_ready_planners();
            let pending = step.take_pending_planner();
            planner = if let Some(io) = pending {
                io.await.unwrap()
            } else {
                ready.pop().unwrap()
            };
            if !morsels.is_empty() {
                for m in morsels {
                    let mut stream = m.into_stream();
                    while let Some(b) = stream.next().await {
                        b.unwrap();
                    }
                }
                break;
            }
        }
        // Exhaust remaining pool capacity externally before driving slice 1.
        let external = MemoryConsumer::new("external").register(&pool);
        external.try_grow(pool_cap - pool.reserved()).unwrap();
        let err =
            tokio::time::timeout(Duration::from_secs(5), drive_planner_collect(planner))
                .await
                .expect("must not deadlock on mid-RG pool exhaustion")
                .unwrap_err();
        assert!(
            matches!(err, DataFusionError::ResourcesExhausted(_)),
            "expected ResourcesExhausted mid-RG, got: {err:?}"
        );
        assert_eq!(split_ctx2.budget.live_bytes(), 0);
        assert_eq!(split_ctx2.budget.admitted_frontier_bytes(), 0);
        drop(external);
        assert_eq!(pool.reserved(), 0);
    }

    #[tokio::test]
    async fn injected_storage_error_mid_rg_reports_context_and_never_replays() {
        let n = 2400usize;
        let (data, _) = build_indexed_dict_fixture(n);
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(64 << 20));
        let metrics = ExecutionPlanMetricsSet::new();
        let cfg = ParquetSplitConfig {
            enabled: true,
            min_row_group_rows: 800,
            target_slice_rows: 400,
            ..Default::default()
        };
        let split_ctx = Arc::new(SplitContext::new(
            cfg,
            Some(Arc::clone(&pool)),
            metrics.clone(),
        ));
        // Slice 0 succeeds (call 0), slice 1 fails (call 1).
        let recorder = RecordingReader::new(data.clone()).with_fail_on_call(1);
        let planner = make_split_plain_planner(
            &data,
            recorder.clone(),
            &[0, 1],
            Arc::clone(&split_ctx),
            &metrics,
        );

        let err = drive_planner_collect(planner)
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("fixture.parquet"), "{err}");
        assert!(err.contains("row_group=0"), "{err}");
        assert!(err.contains("rows="), "{err}");
        assert!(err.contains("injected storage read failure"), "{err}");
        // Only 2 calls happened (slice 0 succeeded, slice 1 failed; no replay).
        assert_eq!(recorder.calls.load(Ordering::SeqCst), 2);
        assert_eq!(split_ctx.budget.live_bytes(), 0);
        assert_eq!(split_ctx.budget.admitted_frontier_bytes(), 0);
        assert_eq!(pool.reserved(), 0);
    }

    #[tokio::test]
    async fn stream_drop_mid_rg_releases_all_reservations() {
        let n = 2400usize;
        let (data, _) = build_indexed_dict_fixture(n);
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(64 << 20));
        let metrics = ExecutionPlanMetricsSet::new();
        let cfg = ParquetSplitConfig {
            enabled: true,
            min_row_group_rows: 800,
            target_slice_rows: 400,
            ..Default::default()
        };
        let split_ctx = Arc::new(SplitContext::new(
            cfg,
            Some(Arc::clone(&pool)),
            metrics.clone(),
        ));
        let mut planner = make_split_plain_planner(
            &data,
            RecordingReader::new(data.clone()),
            &[0, 1],
            Arc::clone(&split_ctx),
            &metrics,
        );

        // Advance until the first slice morsel is published, then drop both
        // the morsel and the remaining planner without draining them.
        let mut first_morsel = None;
        while first_morsel.is_none() {
            let mut step = planner.plan().unwrap().unwrap();
            if let Some(m) = step.take_morsels().pop() {
                first_morsel = Some(m);
            }
            let mut ready = step.take_ready_planners();
            let pending = step.take_pending_planner();
            planner = if let Some(io) = pending {
                io.await.unwrap()
            } else {
                ready.pop().unwrap()
            };
        }
        assert!(split_ctx.budget.live_bytes() > 0);
        assert!(split_ctx.budget.admitted_frontier_bytes() > 0);
        assert!(pool.reserved() > 0);

        drop(first_morsel);
        drop(planner);

        assert_eq!(split_ctx.budget.live_bytes(), 0);
        assert_eq!(split_ctx.budget.admitted_frontier_bytes(), 0);
        assert_eq!(pool.reserved(), 0);
    }

    #[tokio::test]
    async fn async_range_broker_deduplicates_concurrent_in_flight_shared_range() {
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(64 << 20));
        let metrics = ExecutionPlanMetricsSet::new();
        let split_ctx = Arc::new(SplitContext::new(
            ParquetSplitConfig::default(),
            Some(Arc::clone(&pool)),
            metrics,
        ));
        let shared_key = (100u64, 228u64);
        let counts = HashMap::from([(shared_key, 2usize)]);
        let broker = Arc::new(AsyncRangeBroker::new(counts, Arc::clone(&split_ctx)));

        let fetch_calls = Arc::new(AtomicUsize::new(0));
        let started = Arc::new(Barrier::new(2));
        let release_io = Arc::new(Barrier::new(2));

        let b1 = Arc::clone(&broker);
        let calls1 = Arc::clone(&fetch_calls);
        let s1 = Arc::clone(&started);
        let r1 = Arc::clone(&release_io);
        let req_range = 100u64..228u64;
        let req1 = req_range.clone();
        let t1 = SpawnedTask::spawn(async move {
            let mut tracker = Phase2SliceTracker::new(Arc::clone(&b1), vec![shared_key]);
            b1.resolve_ranges(
                std::slice::from_ref(&req1),
                &mut tracker,
                RangeAdmission::Phase2 { phase2_lease: None },
                "slice0",
                move |ranges| {
                    let calls = Arc::clone(&calls1);
                    let s = Arc::clone(&s1);
                    let r = Arc::clone(&r1);
                    async move {
                        calls.fetch_add(1, Ordering::SeqCst);
                        s.wait().await;
                        r.wait().await;
                        Ok(ranges
                            .into_iter()
                            .map(|rng| {
                                Bytes::from(vec![42u8; (rng.end - rng.start) as usize])
                            })
                            .collect())
                    }
                },
            )
            .await
            .unwrap()
        });

        // Wait until t1 has claimed `shared_key` as InFlight and entered I/O.
        started.wait().await;

        let b2 = Arc::clone(&broker);
        let calls2 = Arc::clone(&fetch_calls);
        let t2 = SpawnedTask::spawn(async move {
            let mut tracker = Phase2SliceTracker::new(Arc::clone(&b2), vec![shared_key]);
            b2.resolve_ranges(
                std::slice::from_ref(&req_range),
                &mut tracker,
                RangeAdmission::Phase2 { phase2_lease: None },
                "slice1",
                move |_ranges| {
                    let calls = Arc::clone(&calls2);
                    async move {
                        calls.fetch_add(1, Ordering::SeqCst);
                        panic!("second consumer must deduplicate in-flight fetch")
                    }
                },
            )
            .await
            .unwrap()
        });

        // Give t2 a moment to register on the watch channel, then release t1's I/O.
        tokio::task::yield_now().await;
        release_io.wait().await;

        let res1 = t1.await.unwrap();
        let res2 = t2.await.unwrap();
        assert_eq!(fetch_calls.load(Ordering::SeqCst), 1);
        assert_eq!(res1[0], res2[0]);
        assert_eq!(res1[0].len(), 128);
        // Both views still alive -> live_bytes == 128 (counted ONCE, not 256!)
        assert_eq!(split_ctx.budget.live_bytes(), 128);
        assert_eq!(pool.reserved(), 128);
        drop(res1);
        // One view still alive -> still 128!
        assert_eq!(split_ctx.budget.live_bytes(), 128);
        drop(res2);
        // Final view dropped -> back to 0!
        assert_eq!(split_ctx.budget.live_bytes(), 0);
        assert_eq!(pool.reserved(), 0);
        assert!(broker.state.lock().is_empty());
    }

    /// Backing buffer owner that tracks live allocated bytes returned by
    /// `MergingAllocReader` (mimicking `LocalFileSystem::get_ranges` merging
    /// contiguous ranges into a single allocation).
    struct CountedAlloc {
        buf: Vec<u8>,
        live: Arc<AtomicUsize>,
    }

    impl CountedAlloc {
        fn into_bytes(buf: Vec<u8>, live: Arc<AtomicUsize>) -> Bytes {
            live.fetch_add(buf.len(), Ordering::SeqCst);
            Bytes::from_owner(Self { buf, live })
        }
    }

    impl AsRef<[u8]> for CountedAlloc {
        fn as_ref(&self) -> &[u8] {
            &self.buf
        }
    }

    impl Drop for CountedAlloc {
        fn drop(&mut self) {
            self.live.fetch_sub(self.buf.len(), Ordering::SeqCst);
        }
    }

    #[derive(Clone)]
    struct MergingAllocReader {
        data: Bytes,
        live_allocated: Arc<AtomicUsize>,
        merged_calls: Arc<AtomicUsize>,
    }

    impl MergingAllocReader {
        fn new(data: Bytes) -> Self {
            Self {
                data,
                live_allocated: Arc::new(AtomicUsize::new(0)),
                merged_calls: Arc::new(AtomicUsize::new(0)),
            }
        }

        fn live_allocated(&self) -> usize {
            self.live_allocated.load(Ordering::SeqCst)
        }

        /// Merge contiguous ranges into single `CountedAlloc` buffers and
        /// return slices matching the exact order of `ranges`.
        fn fetch_merged(&self, ranges: &[Range<u64>]) -> Vec<Bytes> {
            if ranges.is_empty() {
                return Vec::new();
            }
            let mut indexed: Vec<(usize, Range<u64>)> =
                ranges.iter().cloned().enumerate().collect();
            indexed.sort_unstable_by_key(|(_, r)| (r.start, r.end));

            struct MergedSpan {
                range: Range<u64>,
                members: Vec<(usize, Range<u64>)>,
            }
            // Group contiguous/overlapping ranges into merged spans.
            let mut spans: Vec<MergedSpan> = Vec::new();
            for (orig_idx, r) in indexed {
                if let Some(last) = spans.last_mut()
                    && r.start <= last.range.end
                {
                    last.range.end = last.range.end.max(r.end);
                    last.members.push((orig_idx, r));
                } else {
                    spans.push(MergedSpan {
                        range: r.clone(),
                        members: vec![(orig_idx, r)],
                    });
                }
            }
            if spans.len() < ranges.len() {
                self.merged_calls.fetch_add(1, Ordering::SeqCst);
            }

            let mut out = vec![Bytes::new(); ranges.len()];
            for span in spans {
                let raw = self.data[span.range.start as usize..span.range.end as usize]
                    .to_vec();
                let merged =
                    CountedAlloc::into_bytes(raw, Arc::clone(&self.live_allocated));
                for (orig_idx, r) in span.members {
                    let lo = (r.start - span.range.start) as usize;
                    let hi = (r.end - span.range.start) as usize;
                    out[orig_idx] = merged.slice(lo..hi);
                }
            }
            out
        }
    }

    impl AsyncFileReader for MergingAllocReader {
        fn get_bytes(
            &mut self,
            range: Range<u64>,
        ) -> BoxFuture<'_, parquet::errors::Result<Bytes>> {
            let mut res = self.fetch_merged(std::slice::from_ref(&range));
            let slice = res.pop().unwrap();
            Box::pin(async move { Ok(slice) })
        }

        fn get_byte_ranges(
            &mut self,
            ranges: Vec<Range<u64>>,
        ) -> BoxFuture<'_, parquet::errors::Result<Vec<Bytes>>> {
            let out = self.fetch_merged(&ranges);
            Box::pin(async move { Ok(out) })
        }

        fn get_metadata<'a>(
            &'a mut self,
            _options: Option<&'a ArrowReaderOptions>,
        ) -> BoxFuture<'a, parquet::errors::Result<Arc<ParquetMetaData>>> {
            let meta = load_metadata(&self.data, true).metadata().clone();
            Box::pin(async move { Ok(meta) })
        }
    }

    #[tokio::test]
    async fn merging_reader_range_broker_live_allocations_bounded_by_budget() {
        let n = 2400usize;
        let (data, _) = build_indexed_dict_fixture(n);
        let reader_metadata = load_metadata(&data, true);
        let file_schema = Arc::clone(reader_metadata.schema());
        let proj_indices = [0usize, 1];
        let output_schema = Arc::new(file_schema.project(&proj_indices).unwrap());
        let projection_exprs = ProjectionExprs::from_indices(&proj_indices, &file_schema);
        let projection = Arc::new(
            DecoderProjection::try_new(
                &projection_exprs,
                &file_schema,
                reader_metadata.parquet_schema(),
                &output_schema,
                None,
            )
            .unwrap(),
        );
        let rg_bytes = reader_metadata.metadata().row_group(0).compressed_size() as u64;
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(64 << 20));
        let metrics = ExecutionPlanMetricsSet::new();
        let cfg = ParquetSplitConfig {
            enabled: true,
            min_row_group_rows: 800,
            target_slice_rows: 400,
            ..Default::default()
        };
        let split_ctx = Arc::new(SplitContext::new(
            cfg,
            Some(Arc::clone(&pool)),
            metrics.clone(),
        ));
        let merging = MergingAllocReader::new(data.clone());
        let pending = vec![(
            RowGroupSelection::new(0, None),
            RgPlanEntry {
                rg_index: 0,
                bytes: rg_bytes,
                fully_matched: false,
            },
        )];
        let readers = BufferedReaders::try_split_plain(SplitPlainBufferedReadersConfig {
            pending,
            reader_metadata,
            projection,
            batch_size: 64,
            force_filter_selections: false,
            enable_page_index: true,
            arrow_reader_metrics: ArrowReaderMetrics::enabled(),
            file_name: Arc::from("merging.parquet"),
            baseline: BaselineMetrics::new(&metrics, 0),
            byte_progress: ByteProgress::new(
                rg_bytes,
                MetricBuilder::new(&metrics).counter("bytes_processed", 0),
            ),
            reader: Box::new(merging.clone()),
            split_context: Arc::clone(&split_ctx),
        })
        .unwrap_or_else(|_| panic!("expected row group to split"));

        let mut planner: Box<dyn MorselPlanner> = Box::new(readers);
        let mut total_rows = 0usize;
        let mut morsels = 0usize;
        while let Some(mut step) = planner.plan().unwrap() {
            for morsel in step.take_morsels() {
                morsels += 1;
                // While the slice reader is live, store allocations must not exceed live_bytes.
                assert!(
                    merging.live_allocated() <= split_ctx.budget.live_bytes(),
                    "while slice {morsels} live: allocated {} > budget {}",
                    merging.live_allocated(),
                    split_ctx.budget.live_bytes()
                );
                let mut stream = morsel.into_stream();
                while let Some(batch) = stream.next().await {
                    total_rows += batch.unwrap().num_rows();
                }
                drop(stream);
                // After the slice reader drops, cached dictionary/boundary entries
                // are detached and MUST NOT keep any merged store allocation alive.
                assert_eq!(
                    merging.live_allocated(),
                    0,
                    "after slice {morsels} dropped: merged store allocation still pinned"
                );
                assert!(
                    merging.live_allocated() <= split_ctx.budget.live_bytes(),
                    "after slice {morsels} dropped: allocated {} > budget {}",
                    merging.live_allocated(),
                    split_ctx.budget.live_bytes()
                );
            }
            let mut ready = step.take_ready_planners();
            let pending = step.take_pending_planner();
            planner = if let Some(io) = pending {
                io.await.unwrap()
            } else if let Some(next) = ready.pop() {
                next
            } else {
                break;
            };
        }

        assert!(morsels >= 4);
        assert_eq!(total_rows, n);
        assert!(
            merging.merged_calls.load(Ordering::SeqCst) > 0,
            "test must exercise contiguous range merging in the reader"
        );
        assert_eq!(merging.live_allocated(), 0);
        assert_eq!(split_ctx.budget.live_bytes(), 0);
        assert_eq!(pool.reserved(), 0);
    }

    #[tokio::test]
    async fn merging_reader_async_range_broker_live_allocations_bounded_by_budget() {
        // Contiguous byte layout:
        // [0..375]    shared dictionary (consumed by slices 0 and 1)
        // [375..2375] private page 0 (slice 0 only)
        // [2375..4375] private page 1 (slice 0 only)
        // [4375..4750] shared boundary page (consumed by slices 0 and 1)
        let data =
            Bytes::from((0..5000usize).map(|i| (i % 251) as u8).collect::<Vec<_>>());
        let merging = MergingAllocReader::new(data.clone());
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(64 << 20));
        let metrics = ExecutionPlanMetricsSet::new();
        let split_ctx = Arc::new(SplitContext::new(
            ParquetSplitConfig {
                enabled: true,
                max_coalesced_request_bytes: 2048,
                ..Default::default()
            },
            Some(Arc::clone(&pool)),
            metrics,
        ));

        let dict_key = (0u64, 375u64);
        let bound_key = (4375u64, 4750u64);
        let counts = HashMap::from([(dict_key, 2usize), (bound_key, 2usize)]);
        let broker = Arc::new(AsyncRangeBroker::new(counts, Arc::clone(&split_ctx)));

        // Slice 0 fetches all four contiguous ranges in one Phase-1/Phase-2 call;
        // `MergingAllocReader` merges all 4750 bytes into a single allocation.
        let mut tracker0 =
            Phase2SliceTracker::new(Arc::clone(&broker), vec![dict_key, bound_key]);
        let req0 = vec![0..375, 375..2375, 2375..4375, 4375..4750];
        let mut admitted0: Option<(Phase2BudgetLease, SplitDecodedLease)> = None;
        let merging_fetch = merging.clone();
        let mut views0 = broker
            .resolve_ranges(
                &req0,
                &mut tracker0,
                RangeAdmission::Phase1 {
                    phase2_candidates: &[],
                    decoded_estimate: 1024,
                    out_leases: &mut admitted0,
                },
                "slice0",
                move |ranges| {
                    let r = merging_fetch.clone();
                    async move { Ok(r.fetch_merged(&ranges)) }
                },
            )
            .await
            .unwrap();
        assert!(
            merging.merged_calls.load(Ordering::SeqCst) > 0,
            "test must merge contiguous ranges into one backing allocation"
        );
        assert_eq!(merging.live_allocated(), 4750);
        // Budget accounts for the full merged call (4750) + detached shared copies (750).
        assert!(
            merging.live_allocated() <= split_ctx.budget.live_bytes(),
            "while slice 0 live: allocated {} > budget {}",
            merging.live_allocated(),
            split_ctx.budget.live_bytes()
        );

        // Drop only the first private view (views0[1]): the second private view
        // (views0[2]) still pins the merged 4750 B allocation, so the shared
        // call lease MUST stay alive and keep allocated <= budget.live_bytes().
        views0[1] = Bytes::new();
        assert_eq!(merging.live_allocated(), 4750);
        assert!(
            merging.live_allocated() <= split_ctx.budget.live_bytes(),
            "after dropping one private view: allocated {} > budget {}",
            merging.live_allocated(),
            split_ctx.budget.live_bytes()
        );

        // Drop all slice 0 views & leases while dict + boundary remain cached
        // as `Ready` for slice 1. The merged 4750 B store allocation MUST drop
        // to 0, leaving only the detached 750 B cached entries accounted.
        drop(views0);
        drop(admitted0);
        drop(tracker0);
        assert_eq!(
            merging.live_allocated(),
            0,
            "cached Ready entries must not pin merged store allocation"
        );
        assert_eq!(split_ctx.budget.live_bytes(), 375 + 375);
        assert!(merging.live_allocated() <= split_ctx.budget.live_bytes());

        // Slice 1 hits both cached Ready entries with zero storage reads.
        let mut tracker1 =
            Phase2SliceTracker::new(Arc::clone(&broker), vec![dict_key, bound_key]);
        let views1 = broker
            .resolve_ranges(
                &[0..375, 4375..4750],
                &mut tracker1,
                RangeAdmission::Phase2 { phase2_lease: None },
                "slice1",
                |_ranges| async move {
                    panic!("slice 1 must hit Ready cache without storage I/O")
                },
            )
            .await
            .unwrap();
        assert_eq!(views1[0].as_ref(), &data[0..375]);
        assert_eq!(views1[1].as_ref(), &data[4375..4750]);
        assert_eq!(merging.live_allocated(), 0);
        assert_eq!(split_ctx.budget.live_bytes(), 375 + 375);

        drop(views1);
        drop(tracker1);
        assert_eq!(merging.live_allocated(), 0);
        assert_eq!(split_ctx.budget.live_bytes(), 0);
        assert_eq!(pool.reserved(), 0);
        assert!(broker.state.lock().is_empty());
    }
}
