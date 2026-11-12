#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
# SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

# Build the DuckDB CLI whose plans are captured (see README.md): DuckDB's
# release build, `GEN=ninja make release`, of a checkout at DUCKDB_COMMIT.
#
# Usage: benchmarks/plan-compare/build_duckdb.sh [DUCKDB_SOURCE_DIR]
#
# DUCKDB_SOURCE_DIR defaults to ~/github/duckdb. If it does not exist, the
# script clones DUCKDB_COMMIT from https://github.com/duckdb/duckdb (shallow,
# like ~/github/duckdb, so that `PRAGMA version` reports the same version). An
# existing checkout must be at DUCKDB_COMMIT without local changes: the script
# does not move it (other users may read it). The CLI is
# DUCKDB_SOURCE_DIR/build/release/duckdb.

set -euo pipefail

# The head of `main` of ~/github/duckdb after `git fetch origin main` when the
# plans were captured (see README.md). capture.py checks that the CLI is built
# from it.
DUCKDB_COMMIT=e55f794cae98443cfaf302f41ce8a83a95a7872a
DUCKDB_REPOSITORY=${DUCKDB_REPOSITORY:-https://github.com/duckdb/duckdb.git}
DUCKDB_SOURCE_DIR=${1:-$HOME/github/duckdb}

if [ ! -e "$DUCKDB_SOURCE_DIR" ]; then
  git init --quiet "$DUCKDB_SOURCE_DIR"
  git -C "$DUCKDB_SOURCE_DIR" remote add origin "$DUCKDB_REPOSITORY"
  git -C "$DUCKDB_SOURCE_DIR" fetch --depth 1 origin "$DUCKDB_COMMIT"
  git -C "$DUCKDB_SOURCE_DIR" checkout --quiet --detach FETCH_HEAD
fi

head=$(git -C "$DUCKDB_SOURCE_DIR" rev-parse HEAD)
if [ "$head" != "$DUCKDB_COMMIT" ]; then
  echo "error: $DUCKDB_SOURCE_DIR is at $head, not at $DUCKDB_COMMIT:" >&2
  echo "  git -C $DUCKDB_SOURCE_DIR fetch origin $DUCKDB_COMMIT" >&2
  echo "  git -C $DUCKDB_SOURCE_DIR checkout $DUCKDB_COMMIT" >&2
  exit 1
fi
if [ -n "$(git -C "$DUCKDB_SOURCE_DIR" status --porcelain --untracked-files=no)" ]; then
  echo "error: $DUCKDB_SOURCE_DIR has local changes" >&2
  exit 1
fi

(cd "$DUCKDB_SOURCE_DIR" && GEN=ninja make release)
"$DUCKDB_SOURCE_DIR/build/release/duckdb" -batch -no-init -no-agent -c "PRAGMA version"
