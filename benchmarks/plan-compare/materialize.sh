#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
# SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

# Materialize the data and the queries of the pipeline engine's benchmark
# (datafusion/physical-pipeline/benches/pipeline_vs_default.rs) for other
# engines (see README.md):
#
# * PIPELINE_BENCH_MODE=data writes the Parquet files of the 8 TPC-H tables
#   and of the `hits` table to DATA_DIR, DATA_DIR/<table>/part-<i>.parquet,
#   one file per table per thread, as the benchmark's `parquet` source does;
# * PIPELINE_BENCH_MODE=queries prints the 22 TPC-H and 25 ClickBench queries,
#   which this script writes to benchmarks/plan-compare/queries.sql.
#
# Usage: benchmarks/plan-compare/materialize.sh [DATA_DIR]
#
# DATA_DIR defaults to benchmarks/plan-compare/data (ignored by git). The
# scale factor, the rows of `hits` and the threads (files per table) are
# PIPELINE_BENCH_SCALE, PIPELINE_BENCH_HITS and PIPELINE_BENCH_THREADS, by
# default those of the captured plans: 0.1, 1000000 and 16 (the defaults of an
# optimized build of the benchmark on a 16-core machine).

set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
DATA_DIR=${1:-$HERE/data}
mkdir -p "$DATA_DIR"
# Absolute, as cargo runs the benchmark in the directory of its package.
DATA_DIR=$(cd "$DATA_DIR" && pwd)

export PIPELINE_BENCH_SCALE=${PIPELINE_BENCH_SCALE:-0.1}
export PIPELINE_BENCH_HITS=${PIPELINE_BENCH_HITS:-1000000}
export PIPELINE_BENCH_THREADS=${PIPELINE_BENCH_THREADS:-16}
# All queries.
unset PIPELINE_BENCH_QUERIES

bench() {
  (cd "$ROOT" && cargo test --profile release-nonlto -p datafusion-physical-pipeline \
    --bench pipeline_vs_default -- pipeline_vs_default --exact)
}

PIPELINE_BENCH_MODE=data PIPELINE_BENCH_DATA_DIR="$DATA_DIR" bench

# Keep the queries of the output only (not the lines of libtest): every query
# is a line `-- <name>` and the lines up to the one that ends with `;`.
PIPELINE_BENCH_MODE=queries bench | awk '
  /^-- [A-Za-z0-9_]+$/ { query = 1; if (count++) print "" }
  query { print }
  query && /;$/ { query = 0 }
' > "$HERE/queries.sql"

count=$(grep -c '^-- ' "$HERE/queries.sql")
if [ "$count" != 47 ]; then
  echo "error: $HERE/queries.sql has $count queries, not 47" >&2
  exit 1
fi
echo "Wrote the data to $DATA_DIR and the $count queries to $HERE/queries.sql"
