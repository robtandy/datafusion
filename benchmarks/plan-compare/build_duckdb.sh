#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
# SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

# Build the DuckDB CLI whose plans are captured (see README.md): DuckDB's
# release build, `GEN=ninja make release`, of a checkout of
# github.com/duckdb/duckdb at DUCKDB_COMMIT.
#
# Usage: benchmarks/plan-compare/build_duckdb.sh [DUCKDB_SOURCE_DIR]
#
# DUCKDB_SOURCE_DIR defaults to ~/github/duckdb-plan-compare. If it does not
# exist, the script clones the history of `main` of
# https://github.com/duckdb/duckdb into it (without the files of other
# commits) and checks out DUCKDB_COMMIT: the version of DuckDB counts the
# commits of its history, which a shallow clone does not have. An existing
# checkout must be at DUCKDB_COMMIT, without local changes: the script does not
# move it, as others may read or build it (~/github/duckdb, which ~/github/dduck
# builds, is at an older commit). The CLI is
# DUCKDB_SOURCE_DIR/build/release/duckdb.

set -euo pipefail

# The head of `main` of github.com/duckdb/duckdb when the plans were captured
# (fetched on 2026-11-12, see README.md). capture.py checks that the CLI is
# built from it.
DUCKDB_COMMIT=5d254ad870763a9724af68b937ebf4e1440ad040
DUCKDB_REPOSITORY=${DUCKDB_REPOSITORY:-https://github.com/duckdb/duckdb.git}
DUCKDB_SOURCE_DIR=${1:-$HOME/github/duckdb-plan-compare}

if [ ! -e "$DUCKDB_SOURCE_DIR" ]; then
  git clone --filter=tree:0 --single-branch --branch main --no-checkout \
    "$DUCKDB_REPOSITORY" "$DUCKDB_SOURCE_DIR"
  git -C "$DUCKDB_SOURCE_DIR" checkout --quiet --detach "$DUCKDB_COMMIT"
fi

head=$(git -C "$DUCKDB_SOURCE_DIR" rev-parse HEAD)
if [ "$head" != "$DUCKDB_COMMIT" ]; then
  echo "error: $DUCKDB_SOURCE_DIR is at $head, not at $DUCKDB_COMMIT:" >&2
  echo "  git -C $DUCKDB_SOURCE_DIR fetch $DUCKDB_REPOSITORY main" >&2
  echo "  git -C $DUCKDB_SOURCE_DIR checkout $DUCKDB_COMMIT" >&2
  exit 1
fi
if [ -n "$(git -C "$DUCKDB_SOURCE_DIR" status --porcelain --untracked-files=no)" ]; then
  echo "error: $DUCKDB_SOURCE_DIR has local changes" >&2
  exit 1
fi

(cd "$DUCKDB_SOURCE_DIR" && GEN=ninja make release)
"$DUCKDB_SOURCE_DIR/build/release/duckdb" -batch -no-init -no-agent -c "PRAGMA version"
