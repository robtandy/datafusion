# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
# SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

import argparse
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import ctypes
from dataclasses import dataclass
import datetime
from decimal import Decimal
import hashlib
import json
import math
import mmap
import os
from pathlib import Path
import platform
import re
import shutil
import stat
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.ipc as pa_ipc
import pyarrow.parquet as pq

BATCH_SIZE = 8192
DEFAULT_SMOKE_ROWS = 50_000
DEFAULT_SMOKE_FILES = 4
DEFAULT_SMOKE_ROW_GROUP_SIZE = 12_500
DEFAULT_SMOKE_SEED = 20261128

QUERIES_DIR = (
    Path(__file__).resolve().parent.parent / "queries" / "clickbench" / "queries"
)
ALL_CLICKBENCH_QUERIES: tuple[str, ...] = tuple(f"q{i}" for i in range(43))

# Official ClickBench Parquet dataset URLs matching benchmarks/bench.sh.
CLICKBENCH_PARTITIONED_URL_TEMPLATE = (
    "https://datasets.clickhouse.com/hits_compatible/athena_partitioned/"
    "hits_{index}.parquet"
)
CLICKBENCH_SINGLE_URL = "https://datasets.clickhouse.com/hits_compatible/hits.parquet"
CLICKBENCH_PARTITIONED_TOTAL_FILES = 100

# Pinned official ClickHouse static release from GitHub Releases.
CLICKHOUSE_RELEASE_TAG = "v26.11.5.8-stable"
CLICKHOUSE_RELEASE_VERSION = "26.11.5.8"
CLICKHOUSE_RELEASE_URL_TEMPLATE = (
    "https://github.com/ClickHouse/ClickHouse/releases/download/"
    "{tag}/clickhouse-common-static-{version}-{arch}.tgz"
)
DEFAULT_CLICKHOUSE_DEST = (
    Path.home() / ".cache" / "morsel-clickbench" / "bin" / "clickhouse"
)

# Exact 105-column Arrow schema of the official ClickBench hits_partitioned Parquet files.
HITS_SCHEMA_FIELDS: tuple[tuple[str, str], ...] = (
    ("WatchID", "int64"),
    ("JavaEnable", "int16"),
    ("Title", "binary"),
    ("GoodEvent", "int16"),
    ("EventTime", "int64"),
    ("EventDate", "uint16"),
    ("CounterID", "int32"),
    ("ClientIP", "int32"),
    ("RegionID", "int32"),
    ("UserID", "int64"),
    ("CounterClass", "int16"),
    ("OS", "int16"),
    ("UserAgent", "int16"),
    ("URL", "binary"),
    ("Referer", "binary"),
    ("IsRefresh", "int16"),
    ("RefererCategoryID", "int16"),
    ("RefererRegionID", "int32"),
    ("URLCategoryID", "int16"),
    ("URLRegionID", "int32"),
    ("ResolutionWidth", "int16"),
    ("ResolutionHeight", "int16"),
    ("ResolutionDepth", "int16"),
    ("FlashMajor", "int16"),
    ("FlashMinor", "int16"),
    ("FlashMinor2", "binary"),
    ("NetMajor", "int16"),
    ("NetMinor", "int16"),
    ("UserAgentMajor", "int16"),
    ("UserAgentMinor", "binary"),
    ("CookieEnable", "int16"),
    ("JavascriptEnable", "int16"),
    ("IsMobile", "int16"),
    ("MobilePhone", "int16"),
    ("MobilePhoneModel", "binary"),
    ("Params", "binary"),
    ("IPNetworkID", "int32"),
    ("TraficSourceID", "int16"),
    ("SearchEngineID", "int16"),
    ("SearchPhrase", "binary"),
    ("AdvEngineID", "int16"),
    ("IsArtifical", "int16"),
    ("WindowClientWidth", "int16"),
    ("WindowClientHeight", "int16"),
    ("ClientTimeZone", "int16"),
    ("ClientEventTime", "int64"),
    ("SilverlightVersion1", "int16"),
    ("SilverlightVersion2", "int16"),
    ("SilverlightVersion3", "int32"),
    ("SilverlightVersion4", "int16"),
    ("PageCharset", "binary"),
    ("CodeVersion", "int32"),
    ("IsLink", "int16"),
    ("IsDownload", "int16"),
    ("IsNotBounce", "int16"),
    ("FUniqID", "int64"),
    ("OriginalURL", "binary"),
    ("HID", "int32"),
    ("IsOldCounter", "int16"),
    ("IsEvent", "int16"),
    ("IsParameter", "int16"),
    ("DontCountHits", "int16"),
    ("WithHash", "int16"),
    ("HitColor", "binary"),
    ("LocalEventTime", "int64"),
    ("Age", "int16"),
    ("Sex", "int16"),
    ("Income", "int16"),
    ("Interests", "int16"),
    ("Robotness", "int16"),
    ("RemoteIP", "int32"),
    ("WindowName", "int32"),
    ("OpenerName", "int32"),
    ("HistoryLength", "int16"),
    ("BrowserLanguage", "binary"),
    ("BrowserCountry", "binary"),
    ("SocialNetwork", "binary"),
    ("SocialAction", "binary"),
    ("HTTPError", "int16"),
    ("SendTiming", "int32"),
    ("DNSTiming", "int32"),
    ("ConnectTiming", "int32"),
    ("ResponseStartTiming", "int32"),
    ("ResponseEndTiming", "int32"),
    ("FetchTiming", "int32"),
    ("SocialSourceNetworkID", "int16"),
    ("SocialSourcePage", "binary"),
    ("ParamPrice", "int64"),
    ("ParamOrderID", "binary"),
    ("ParamCurrency", "binary"),
    ("ParamCurrencyID", "int16"),
    ("OpenstatServiceName", "binary"),
    ("OpenstatCampaignID", "binary"),
    ("OpenstatAdID", "binary"),
    ("OpenstatSourceID", "binary"),
    ("UTMSource", "binary"),
    ("UTMMedium", "binary"),
    ("UTMCampaign", "binary"),
    ("UTMContent", "binary"),
    ("UTMTerm", "binary"),
    ("FromTag", "binary"),
    ("HasGCLID", "int16"),
    ("RefererHash", "int64"),
    ("URLHash", "int64"),
    ("CLID", "int32"),
)

_PA_TYPES = {
    "int16": pa.int16(),
    "uint16": pa.uint16(),
    "int32": pa.int32(),
    "int64": pa.int64(),
    "binary": pa.binary(),
}


def hits_arrow_schema() -> pa.Schema:
    return pa.schema(
        [pa.field(name, _PA_TYPES[typ]) for name, typ in HITS_SCHEMA_FIELDS]
    )


@dataclass(frozen=True)
class QuerySpec:
    name: str
    index: int
    sql: str
    order_mode: str  # "exact" | "unordered" | "ordered_ties" | "topn" | "limit_unordered" | "topn_hidden_key"
    sort_key_cols: tuple[int, ...] = ()
    has_offset: bool = False
    limit: int | None = None
    group_key_cols: tuple[int, ...] = ()


# Ordering metadata for all 43 official ClickBench queries:
# (order_mode, sort_key_cols, has_offset, limit, group_key_cols)
_QUERY_ORDER_SPECS: dict[
    int, tuple[str, tuple[int, ...], bool, int | None, tuple[int, ...]]
] = {
    0: ("exact", (), False, None, ()),
    1: ("exact", (), False, None, ()),
    2: ("exact", (), False, None, ()),
    3: ("exact", (), False, None, ()),
    4: ("exact", (), False, None, ()),
    5: ("exact", (), False, None, ()),
    6: ("exact", (), False, None, ()),
    7: ("ordered_ties", (1,), False, None, (0,)),
    8: ("topn", (1,), False, 10, (0,)),
    9: ("topn", (2,), False, 10, (0,)),
    10: ("topn", (1,), False, 10, (0,)),
    11: ("topn", (2,), False, 10, (0, 1)),
    12: ("topn", (1,), False, 10, (0,)),
    13: ("topn", (1,), False, 10, (0,)),
    14: ("topn", (2,), False, 10, (0, 1)),
    15: ("topn", (1,), False, 10, (0,)),
    16: ("topn", (2,), False, 10, (0, 1)),
    17: ("limit_unordered", (), False, 10, (0, 1)),
    18: ("topn", (3,), False, 10, (0, 1, 2)),
    19: ("unordered", (), False, None, ()),
    20: ("exact", (), False, None, ()),
    21: ("topn", (2,), False, 10, (0,)),
    22: ("topn", (3,), False, 10, (0,)),
    23: ("topn", (4,), False, 10, ()),
    24: ("topn_hidden_key", (), False, 10, ()),
    25: ("exact", (), False, 10, ()),
    26: ("exact", (), False, 10, ()),
    27: ("topn", (1,), False, 25, (0,)),
    28: ("topn", (1,), False, 25, (0,)),
    29: ("exact", (), False, None, ()),
    30: ("topn", (2,), False, 10, (0, 1)),
    31: ("topn", (2,), False, 10, (0, 1)),
    32: ("topn", (2,), False, 10, (0, 1)),
    33: ("topn", (1,), False, 10, (0,)),
    34: ("topn", (2,), False, 10, (0, 1)),
    35: ("topn", (4,), False, 10, (0, 1, 2, 3)),
    36: ("topn", (1,), False, 10, (0,)),
    37: ("topn", (1,), False, 10, (0,)),
    38: ("topn", (1,), True, 10, (0,)),
    39: ("topn", (5,), True, 10, (0, 1, 2, 3, 4)),
    40: ("topn", (2,), True, 10, (0, 1)),
    41: ("topn", (2,), True, 10, (0, 1)),
    42: ("exact", (), True, 10, (0,)),
}


def load_clickbench_query(query_num: int, queries_dir: Path = QUERIES_DIR) -> str:
    """Load an official ClickBench query from benchmarks/queries/clickbench/queries/q<N>.sql."""
    sql_path = queries_dir / f"q{query_num}.sql"
    lines = [
        line
        for line in sql_path.read_text().splitlines()
        if not line.lstrip().startswith("--")
    ]
    sql = "\n".join(lines).strip().rstrip(";").strip()
    if not sql:
        raise ValueError(f"empty SQL in {sql_path}")
    return sql


def _build_queries() -> dict[str, QuerySpec]:
    out: dict[str, QuerySpec] = {}
    for idx in range(43):
        mode, key_cols, has_offset, limit, group_key_cols = _QUERY_ORDER_SPECS[idx]
        name = f"q{idx}"
        out[name] = QuerySpec(
            name=name,
            index=idx,
            sql=load_clickbench_query(idx),
            order_mode=mode,
            sort_key_cols=key_cols,
            has_offset=has_offset,
            limit=limit,
            group_key_cols=group_key_cols,
        )
    return out


QUERIES: dict[str, QuerySpec] = _build_queries()


# Explicit, documented per-query dialect adaptations for DuckDB and ClickHouse.
DUCKDB_DIALECT_NOTES: dict[str, str] = {
    "q18": (
        'to_timestamp_seconds("EventTime") -> to_timestamp("EventTime") '
        "with SET TimeZone = 'UTC'"
    ),
    "q27": 'octet_length("URL") -> strlen("URL") (UTF-8 byte length on VARCHAR)',
    "q28": 'octet_length("Referer") -> strlen("Referer") (UTF-8 byte length on VARCHAR)',
    "q42": (
        'to_timestamp_seconds("EventTime") -> CAST(to_timestamp("EventTime") AS TIMESTAMP) '
        "with SET TimeZone = 'UTC'"
    ),
}

CLICKHOUSE_DIALECT_NOTES: dict[str, str] = {
    "q3": (
        'AVG("UserID") -> AVG(toFloat64("UserID")) '
        "(ClickHouse avg(Int64) accumulates in 64-bit integer and silently overflows "
        "on 64-bit UserIDs; casting to Float64 matches DataFusion and DuckDB)"
    ),
    "q18": ('to_timestamp_seconds("EventTime") -> toDateTime("EventTime", \'UTC\')'),
    "q28": (
        "Add RE2 (?-s) flag in REGEXP_REPLACE pattern so '.' does not match '\\n' "
        "(ClickHouse RE2 defaults to dot_nl=true, while DataFusion Rust regex and "
        "DuckDB RE2 default to dot_nl=false)"
    ),
    "q42": (
        'to_timestamp_seconds("EventTime") -> toDateTime64("EventTime", 0, \'UTC\') '
        "(ClickHouse ArrowStream encodes DateTime as uint32 epoch seconds, whereas "
        "DateTime64(0, 'UTC') exports as Arrow timestamp[s, tz=UTC] matching DataFusion/DuckDB)"
    ),
}


def adapt_sql_for_duckdb(query_name: str, sql: str) -> str:
    if query_name == "q18":
        return sql.replace(
            'to_timestamp_seconds("EventTime")', 'to_timestamp("EventTime")'
        )
    if query_name == "q27":
        return sql.replace('octet_length("URL")', 'strlen("URL")')
    if query_name == "q28":
        return sql.replace('octet_length("Referer")', 'strlen("Referer")')
    if query_name == "q42":
        return sql.replace(
            'to_timestamp_seconds("EventTime")',
            'CAST(to_timestamp("EventTime") AS TIMESTAMP)',
        )
    return sql


def adapt_sql_for_clickhouse(query_name: str, sql: str) -> str:
    if query_name == "q3":
        return sql.replace('AVG("UserID")', 'AVG(toFloat64("UserID"))')
    if query_name == "q18":
        return sql.replace(
            'to_timestamp_seconds("EventTime")',
            "toDateTime(\"EventTime\", 'UTC')",
        )
    if query_name == "q42":
        return sql.replace(
            'to_timestamp_seconds("EventTime")',
            "toDateTime64(\"EventTime\", 0, 'UTC')",
        )
    if query_name == "q28":
        return sql.replace(
            "'^https?://(?:www\\.)?([^/]+)/.*$'",
            "'(?-s)^https?://(?:www\\.)?([^/]+)/.*$'",
        )
    return sql


@dataclass(frozen=True)
class Arm:
    name: str
    engine: str  # "datafusion" | "duckdb" | "clickhouse"
    mode: str | None  # "off" | "required" | None
    profile: str | None


ARMS: dict[str, Arm] = {
    "stock_default": Arm("stock_default", "datafusion", "off", "default"),
    "stock_pushdown": Arm("stock_pushdown", "datafusion", "off", "pushdown"),
    "stock_pushdown_no_repartition": Arm(
        "stock_pushdown_no_repartition",
        "datafusion",
        "off",
        "pushdown_no_repartition",
    ),
    "morsel_pushdown": Arm("morsel_pushdown", "datafusion", "required", "pushdown"),
    "morsel_pruning": Arm("morsel_pruning", "datafusion", "required", "pruning"),
    "morsel_no_pushdown": Arm(
        "morsel_no_pushdown", "datafusion", "required", "no_pushdown"
    ),
    "duckdb_default": Arm("duckdb_default", "duckdb", None, None),
    "clickhouse_default": Arm("clickhouse_default", "clickhouse", None, None),
}

DEFAULT_BENCH_ARMS: tuple[str, ...] = (
    "stock_default",
    "stock_pushdown",
    "morsel_pushdown",
    "duckdb_default",
    "clickhouse_default",
)


def resolve_queries(raw: list[str] | None) -> list[str]:
    if not raw:
        return list(ALL_CLICKBENCH_QUERIES)
    names: list[str] = []
    for item in raw:
        for token in item.split(","):
            tok = token.strip().lower()
            if not tok:
                continue
            if tok == "all":
                names.extend(ALL_CLICKBENCH_QUERIES)
                continue
            if "-" in tok and tok.startswith("q"):
                lo_s, hi_s = tok.split("-", 1)
                lo = int(lo_s.lstrip("q"))
                hi = int(hi_s.lstrip("q"))
                if lo > hi or lo < 0 or hi > 42:
                    raise ValueError(f"invalid ClickBench query range: '{tok}'")
                names.extend(f"q{i}" for i in range(lo, hi + 1))
                continue
            if tok not in QUERIES:
                raise ValueError(
                    f"unknown ClickBench query '{tok}'; expected q0..q42 or 'all'"
                )
            names.append(tok)
    if not names:
        raise ValueError("empty query list")
    # Preserve order while deduplicating.
    return list(dict.fromkeys(names))


def resolve_arms(raw: list[str] | None, default: tuple[str, ...]) -> list[str]:
    if not raw:
        return list(default)
    names: list[str] = []
    for item in raw:
        for token in item.split(","):
            tok = token.strip()
            if not tok:
                continue
            if tok not in ARMS:
                raise ValueError(
                    f"unknown arm '{tok}'; expected one of {', '.join(ARMS)}"
                )
            names.append(tok)
    if not names:
        raise ValueError("empty arm list")
    return list(dict.fromkeys(names))


def _parquet_sort_key(path: Path) -> tuple[int, int, str]:
    match = re.fullmatch(r"hits_(\d+)\.parquet", path.name)
    if match:
        return (0, int(match.group(1)), path.name)
    return (1, 0, path.name)


def dataset_files(path: Path) -> list[Path]:
    """Return sorted Parquet files for either a single file or a partitioned directory."""
    if path.is_file():
        if path.suffix != ".parquet":
            raise ValueError(f"expected a .parquet file: {path}")
        return [path]
    if path.is_dir():
        files = sorted(
            (p for p in path.iterdir() if p.is_file() and p.suffix == ".parquet"),
            key=_parquet_sort_key,
        )
        if not files:
            raise ValueError(f"no .parquet files found in directory: {path}")
        return files
    raise ValueError(f"dataset path does not exist: {path}")


def dataset_summary(path: Path) -> dict:
    files = dataset_files(path)
    metas = [pq.read_metadata(f) for f in files]
    schema = pq.read_schema(files[0])
    return {
        "path": str(path.resolve()),
        "layout": "single_file" if path.is_file() else "partitioned",
        "files": len(files),
        "columns": len(schema),
        "rows": sum(m.num_rows for m in metas),
        "row_groups": sum(m.num_row_groups for m in metas),
        "parquet_bytes": sum(f.stat().st_size for f in files),
        "first_file": files[0].name,
        "last_file": files[-1].name,
    }


def _weighted_pool(rng: np.random.Generator, items: list, rows: int) -> list:
    n = len(items)
    # Zipf-like weights produce distinct frequencies across items to minimize tie noise.
    ranks = np.arange(1, n + 1, dtype=np.float64)
    probs = 1.0 / np.power(ranks, 1.15)
    probs /= probs.sum()
    idx = rng.choice(n, size=rows, p=probs)
    return [items[i] for i in idx]


def generate_smoke(
    path: Path,
    rows: int = DEFAULT_SMOKE_ROWS,
    files: int = DEFAULT_SMOKE_FILES,
    row_group_size: int = DEFAULT_SMOKE_ROW_GROUP_SIZE,
    seed: int = DEFAULT_SMOKE_SEED,
) -> dict:
    """Deterministically generate a ClickBench-schema partitioned Parquet smoke dataset."""
    if path.exists():
        raise FileExistsError(f"output directory already exists: {path}")
    if rows < 1:
        raise ValueError("rows must be positive")
    if files < 1 or files > rows:
        raise ValueError("files must be between 1 and rows")

    path.mkdir(parents=True)
    rng = np.random.default_rng(seed)

    # EventDate: uint16 days since 1970-01-01.
    # 2013-07-01 = 15887, 2013-07-14 = 15900, 2013-07-15 = 15901, 2013-07-31 = 15917.
    # Weight 2013-07-14 and 2013-07-15 heavily so q42 (OFFSET 1000) returns rows.
    july_days = np.arange(15887, 15918, dtype=np.uint16)
    day_probs = np.full(len(july_days), 1.0, dtype=np.float64)
    day_probs[15900 - 15887] = 12.0  # 2013-07-14
    day_probs[15901 - 15887] = 12.0  # 2013-07-15
    day_probs /= day_probs.sum()
    event_date = rng.choice(july_days, size=rows, p=day_probs).astype(np.uint16)

    # EventTime: unix seconds consistent with EventDate, spanning all 1440 minutes/day.
    # Use unique sub-second-free offsets where possible while exercising all minutes.
    sec_in_day = rng.integers(0, 86_400, size=rows, dtype=np.int64)
    event_time = event_date.astype(np.int64) * 86_400 + sec_in_day

    # CounterID: ~40% CounterID=62 (for q36..q42), rest skewed across 20 IDs.
    counter_pool = np.array(
        [62] + [1000 + i * 37 for i in range(1, 25)], dtype=np.int32
    )
    counter_probs = np.array(
        [0.40] + [0.60 / (i**1.15) for i in range(1, 25)], dtype=np.float64
    )
    counter_probs[1:] *= 0.60 / counter_probs[1:].sum()
    counter_id = rng.choice(counter_pool, size=rows, p=counter_probs).astype(np.int32)

    # AdvEngineID: ~65% zero, ~35% skewed 1..12.
    adv_pool = np.arange(0, 13, dtype=np.int16)
    adv_probs = np.array(
        [0.65] + [0.35 / (i**1.2) for i in range(1, 13)], dtype=np.float64
    )
    adv_probs[1:] *= 0.35 / adv_probs[1:].sum()
    adv_engine_id = rng.choice(adv_pool, size=rows, p=adv_probs).astype(np.int16)

    # RegionID: skewed across 40 regions.
    region_pool = np.array([10 + i * 17 for i in range(40)], dtype=np.int32)
    region_probs = 1.0 / np.power(np.arange(1, 41, dtype=np.float64), 1.1)
    region_probs /= region_probs.sum()
    region_id = rng.choice(region_pool, size=rows, p=region_probs).astype(np.int32)

    # UserID: include target 435090932899640449 (q19) + 1500 skewed users.
    user_pool = np.array(
        [435090932899640449] + [100_000_000_000 + i * 97_531 for i in range(1500)],
        dtype=np.int64,
    )
    user_probs = np.full(len(user_pool), 1.0, dtype=np.float64)
    user_probs[0] = 0.0006
    tail = 1.0 / np.power(np.arange(1, 1501, dtype=np.float64), 0.95)
    user_probs[1:] = (1.0 - user_probs[0]) * (tail / tail.sum())
    user_id = rng.choice(user_pool, size=rows, p=user_probs).astype(np.int64)
    # Guarantee at least 3 rows match q19 even for small --rows.
    if rows >= 10:
        user_id[0] = 435090932899640449
        user_id[rows // 2] = 435090932899640449
        user_id[-1] = 435090932899640449

    # WatchID: mostly unique int64 with a small repeating hot set for q31/q32.
    watch_id = rng.integers(1_000_000_000, 9_000_000_000, size=rows, dtype=np.int64)
    hot_watches = np.array([500_000_000 + i for i in range(25)], dtype=np.int64)
    hot_mask = rng.random(rows) < 0.03
    watch_id[hot_mask] = rng.choice(hot_watches, size=int(hot_mask.sum()))

    client_ip = rng.choice(
        np.array([10_000_000 + i * 257 for i in range(300)], dtype=np.int32),
        size=rows,
        p=(
            (1.0 / np.power(np.arange(1, 301, dtype=np.float64), 1.1))
            / np.sum(1.0 / np.power(np.arange(1, 301, dtype=np.float64), 1.1))
        ),
    ).astype(np.int32)

    res_width = rng.choice(
        np.array([1024, 1280, 1366, 1440, 1600, 1920, 2560], dtype=np.int16),
        size=rows,
    ).astype(np.int16)
    res_height = rng.choice(
        np.array([768, 800, 900, 1024, 1080, 1440], dtype=np.int16),
        size=rows,
    ).astype(np.int16)

    is_refresh = (rng.random(rows) < 0.12).astype(np.int16)
    dont_count_hits = (rng.random(rows) < 0.10).astype(np.int16)
    is_link = (rng.random(rows) < 0.55).astype(np.int16)
    is_download = (rng.random(rows) < 0.08).astype(np.int16)

    trafic_source_id = rng.choice(
        np.array([-1, 0, 1, 2, 3, 6], dtype=np.int16),
        size=rows,
        p=[0.25, 0.15, 0.15, 0.10, 0.10, 0.25],
    ).astype(np.int16)
    search_engine_id = rng.choice(
        np.array([0, 1, 2, 3, 4, 5], dtype=np.int16),
        size=rows,
        p=[0.50, 0.18, 0.12, 0.09, 0.06, 0.05],
    ).astype(np.int16)
    mobile_phone = rng.choice(
        np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.int16),
        size=rows,
        p=[0.55, 0.14, 0.10, 0.08, 0.06, 0.04, 0.03],
    ).astype(np.int16)

    mobile_models = [b""] + [f"Model-{i:02d}".encode() for i in range(1, 25)]
    model_probs = np.array(
        [0.55] + [0.45 / (i**1.15) for i in range(1, 25)], dtype=np.float64
    )
    model_probs[1:] *= 0.45 / model_probs[1:].sum()
    mobile_model_idx = rng.choice(len(mobile_models), size=rows, p=model_probs)
    mobile_phone_model = [mobile_models[i] for i in mobile_model_idx]

    # SearchPhrase: ~45% empty, ~55% skewed across 300 UTF-8 phrases (including Cyrillic).
    phrases = [b""] + [
        (f"поиск запрос {i}" if i % 5 == 0 else f"clickbench phrase {i:03d}").encode(
            "utf-8"
        )
        for i in range(1, 301)
    ]
    phrase_probs = np.array(
        [0.45] + [0.55 / (i**1.05) for i in range(1, 301)], dtype=np.float64
    )
    phrase_probs[1:] *= 0.55 / phrase_probs[1:].sum()
    phrase_idx = rng.choice(len(phrases), size=rows, p=phrase_probs)
    search_phrase = [phrases[i] for i in phrase_idx]

    # URL: >1,200 distinct URLs including google URLs, non-.google. URLs, and multi-byte UTF-8.
    url_pool: list[bytes] = [b""]
    for i in range(1, 1301):
        if i <= 30:
            url_pool.append(f"https://www.google.com/search?q=item_{i}".encode("utf-8"))
        elif i <= 60:
            url_pool.append(
                f"http://mygoogle-news.org/статья/{i}/index.html".encode("utf-8")
            )
        elif i % 7 == 0:
            url_pool.append(
                f"https://shop.example.ru/товар/{i}?ref=click".encode("utf-8")
            )
        else:
            url_pool.append(
                f"https://site{i % 40}.example.com/page/{i}?v=1".encode("utf-8")
            )
    url_probs = np.array(
        [0.05] + [0.95 / (i**0.75) for i in range(1, len(url_pool))], dtype=np.float64
    )
    url_probs[1:] *= 0.95 / url_probs[1:].sum()
    url_idx = rng.choice(len(url_pool), size=rows, p=url_probs)
    urls = [url_pool[i] for i in url_idx]

    # Title: mix of empty, 'Google ...' titles (exercises q22), and UTF-8 titles.
    title_pool = [b""] + [
        (
            f"Google News Item {i}"
            if i <= 20
            else (f"Заголовок страницы {i}" if i % 4 == 0 else f"Page Title {i:03d}")
        ).encode("utf-8")
        for i in range(1, 121)
    ]
    title_probs = np.array(
        [0.10] + [0.90 / (i**1.05) for i in range(1, len(title_pool))], dtype=np.float64
    )
    title_probs[1:] *= 0.90 / title_probs[1:].sum()
    title_idx = rng.choice(len(title_pool), size=rows, p=title_probs)
    titles = [title_pool[i] for i in title_idx]

    # Referer: mix of empty, standard URLs, multi-byte UTF-8 URLs, newline URLs
    # with distinct byte lengths (so q28's AVG(octet_length) ranks them without
    # ties and detects ClickHouse RE2 dot_nl differences), and non-matching strings.
    ref_pool = [b""]
    for i in range(1, 201):
        if i <= 10:
            pad = "p" * (i * 2)
            ref_pool.append(
                f"https://www.refhost{i}.com/search/{pad}/item_{i}?a=1".encode("utf-8")
            )
        elif i <= 15:
            # Distinct lengths and paired non-newline / newline URLs for the same
            # host exercise ClickHouse RE2 (?-s) parity on smoke-sized q28.
            pad = "x" * ((i - 10) * 6)
            ref_pool.append(
                f"http://nlhost{i}.org/path/{pad}\nsecond_line_{i}".encode("utf-8")
            )
        elif i <= 20:
            host_idx = i - 5
            pad = "a" * ((host_idx - 10) * 3)
            ref_pool.append(f"http://nlhost{host_idx}.org/base/{pad}".encode("utf-8"))
        elif i <= 30:
            pad = "у" * (i - 20)
            ref_pool.append(
                f"http://ruhost{i}.ru/поиск/{pad}/{i}/страница".encode("utf-8")
            )
        elif i <= 40:
            ref_pool.append(f"non-url-referer-{i}-{'z' * (i - 30)}".encode("utf-8"))
        else:
            ref_pool.append(f"http://host{i}.net/p/{i}/details".encode("utf-8"))
    ref_probs = np.array(
        [0.20] + [0.80 / (i**0.9) for i in range(1, len(ref_pool))], dtype=np.float64
    )
    ref_probs[1:] *= 0.80 / ref_probs[1:].sum()
    ref_idx = rng.choice(len(ref_pool), size=rows, p=ref_probs)
    referers = [ref_pool[i] for i in ref_idx]

    # RefererHash & URLHash: include the exact q40 / q41 constants.
    ref_hash_pool = np.array(
        [3594120000172545465] + [10_000_000_000 + i * 1_009 for i in range(1, 80)],
        dtype=np.int64,
    )
    ref_hash_probs = np.array(
        [0.15] + [0.85 / 79.0] * 79,
        dtype=np.float64,
    )
    referer_hash = rng.choice(ref_hash_pool, size=rows, p=ref_hash_probs).astype(
        np.int64
    )

    url_hash_pool = np.array(
        [2868770270353813622] + [20_000_000_000 + i * 2_017 for i in range(1, 200)],
        dtype=np.int64,
    )
    url_hash_probs = np.array(
        [0.12] + [0.88 / 199.0] * 199,
        dtype=np.float64,
    )
    url_hash = rng.choice(url_hash_pool, size=rows, p=url_hash_probs).astype(np.int64)

    win_width = rng.integers(320, 2560, size=rows, dtype=np.int16)
    win_height = rng.integers(240, 1440, size=rows, dtype=np.int16)

    # Assemble all 105 columns matching HITS_SCHEMA_FIELDS exactly.
    col_data: dict[str, pa.Array] = {
        "WatchID": pa.array(watch_id, type=pa.int64()),
        "Title": pa.array(titles, type=pa.binary()),
        "EventTime": pa.array(event_time, type=pa.int64()),
        "EventDate": pa.array(event_date, type=pa.uint16()),
        "CounterID": pa.array(counter_id, type=pa.int32()),
        "ClientIP": pa.array(client_ip, type=pa.int32()),
        "RegionID": pa.array(region_id, type=pa.int32()),
        "UserID": pa.array(user_id, type=pa.int64()),
        "URL": pa.array(urls, type=pa.binary()),
        "Referer": pa.array(referers, type=pa.binary()),
        "IsRefresh": pa.array(is_refresh, type=pa.int16()),
        "ResolutionWidth": pa.array(res_width, type=pa.int16()),
        "ResolutionHeight": pa.array(res_height, type=pa.int16()),
        "MobilePhone": pa.array(mobile_phone, type=pa.int16()),
        "MobilePhoneModel": pa.array(mobile_phone_model, type=pa.binary()),
        "TraficSourceID": pa.array(trafic_source_id, type=pa.int16()),
        "SearchEngineID": pa.array(search_engine_id, type=pa.int16()),
        "SearchPhrase": pa.array(search_phrase, type=pa.binary()),
        "AdvEngineID": pa.array(adv_engine_id, type=pa.int16()),
        "WindowClientWidth": pa.array(win_width, type=pa.int16()),
        "WindowClientHeight": pa.array(win_height, type=pa.int16()),
        "ClientEventTime": pa.array(event_time, type=pa.int64()),
        "IsLink": pa.array(is_link, type=pa.int16()),
        "IsDownload": pa.array(is_download, type=pa.int16()),
        "DontCountHits": pa.array(dont_count_hits, type=pa.int16()),
        "LocalEventTime": pa.array(event_time, type=pa.int64()),
        "RefererHash": pa.array(referer_hash, type=pa.int64()),
        "URLHash": pa.array(url_hash, type=pa.int64()),
    }

    zero_i16 = pa.array(np.zeros(rows, dtype=np.int16), type=pa.int16())
    zero_i32 = pa.array(np.zeros(rows, dtype=np.int32), type=pa.int32())
    zero_i64 = pa.array(np.zeros(rows, dtype=np.int64), type=pa.int64())
    empty_bin = pa.array([b""] * rows, type=pa.binary())
    defaults = {
        "int16": zero_i16,
        "int32": zero_i32,
        "int64": zero_i64,
        "binary": empty_bin,
    }

    schema = hits_arrow_schema()
    arrays = [
        col_data[name] if name in col_data else defaults[typ]
        for name, typ in HITS_SCHEMA_FIELDS
    ]
    full_table = pa.Table.from_arrays(arrays, schema=schema)

    # Split deterministically across `files` shards: hits_0.parquet .. hits_{files-1}.parquet.
    base, rem = divmod(rows, files)
    offset = 0
    for idx in range(files):
        shard_len = base + (1 if idx < rem else 0)
        shard = full_table.slice(offset, shard_len)
        offset += shard_len
        out_file = path / f"hits_{idx}.parquet"
        with pq.ParquetWriter(
            out_file,
            schema,
            compression="zstd",
            compression_level=3,
            use_dictionary=True,
            write_statistics=True,
            version="2.6",
            data_page_version="1.0",
        ) as writer:
            writer.write_table(shard, row_group_size=row_group_size)

    return dataset_summary(path)


def _download_one_file(url: str, dest: Path, timeout: int = 300) -> tuple[str, int]:
    """Download a single Parquet file atomically, skipping if already valid."""
    if dest.is_file() and dest.stat().st_size > 0:
        try:
            pq.read_metadata(dest)
            return ("cached", dest.stat().st_size)
        except Exception:
            dest.unlink(missing_ok=True)
    tmp = dest.with_suffix(".parquet.part")
    req = urllib.request.Request(url, headers={"User-Agent": "morsel-clickbench/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, tmp.open("wb") as out:
        shutil.copyfileobj(resp, out, length=1 << 20)
    # Verify Parquet footer before moving into place.
    pq.read_metadata(tmp)
    tmp.replace(dest)
    return ("downloaded", dest.stat().st_size)


def download_dataset(
    dest: Path,
    count: int = CLICKBENCH_PARTITIONED_TOTAL_FILES,
    single: bool = False,
    jobs: int = 8,
    timeout: int = 300,
) -> dict:
    """Download official ClickBench Parquet files outside Git."""
    if single:
        dest.parent.mkdir(parents=True, exist_ok=True)
        out_path = dest if dest.suffix == ".parquet" else dest / "hits.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _download_one_file(CLICKBENCH_SINGLE_URL, out_path, timeout=timeout)
        return dataset_summary(out_path)

    if count < 1 or count > CLICKBENCH_PARTITIONED_TOTAL_FILES:
        raise ValueError(
            f"count must be between 1 and {CLICKBENCH_PARTITIONED_TOTAL_FILES}"
        )
    dest.mkdir(parents=True, exist_ok=True)
    tasks = [
        (
            CLICKBENCH_PARTITIONED_URL_TEMPLATE.format(index=i),
            dest / f"hits_{i}.parquet",
        )
        for i in range(count)
    ]
    downloaded = 0
    cached = 0
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futs = {
            pool.submit(_download_one_file, url, path, timeout): path.name
            for url, path in tasks
        }
        for fut in as_completed(futs):
            status, _ = fut.result()
            if status == "cached":
                cached += 1
            else:
                downloaded += 1
    summary = dataset_summary(dest)
    summary["downloaded_files"] = downloaded
    summary["cached_files"] = cached
    return summary


def clickhouse_arch() -> str:
    machine = platform.machine().lower()
    if machine in ("aarch64", "arm64"):
        return "arm64"
    if machine in ("x86_64", "amd64"):
        return "amd64"
    raise ValueError(
        f"unsupported architecture for ClickHouse static release: {machine}"
    )


def install_clickhouse(
    dest: Path = DEFAULT_CLICKHOUSE_DEST,
    tag: str = CLICKHOUSE_RELEASE_TAG,
    version: str = CLICKHOUSE_RELEASE_VERSION,
    timeout: int = 300,
) -> dict:
    """Download, SHA-512 verify, and install the pinned official ClickHouse static binary."""
    if platform.system() != "Linux":
        raise ValueError("install-clickhouse supports Linux static binaries only")
    arch = clickhouse_arch()
    tgz_url = CLICKHOUSE_RELEASE_URL_TEMPLATE.format(
        tag=tag, version=version, arch=arch
    )
    sha_url = f"{tgz_url}.sha512"

    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="morsel-ch-install-") as tmp_dir:
        tmp = Path(tmp_dir)
        sha_text = (
            urllib.request.urlopen(sha_url, timeout=60).read().decode("utf-8").strip()
        )
        expected_sha = sha_text.split()[0].lower()
        if len(expected_sha) != 128 or not re.fullmatch(r"[0-9a-f]{128}", expected_sha):
            raise ValueError(f"unexpected sha512 format from {sha_url}: {sha_text!r}")

        tgz_path = tmp / f"clickhouse-common-static-{version}-{arch}.tgz"
        req = urllib.request.Request(
            tgz_url, headers={"User-Agent": "morsel-clickbench/1.0"}
        )
        hasher = hashlib.sha512()
        with urllib.request.urlopen(req, timeout=timeout) as resp, tgz_path.open(
            "wb"
        ) as out:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                hasher.update(chunk)
        actual_sha = hasher.hexdigest().lower()
        if actual_sha != expected_sha:
            raise ValueError(
                f"SHA-512 mismatch for {tgz_url}: expected {expected_sha}, got {actual_sha}"
            )

        extracted_bin = tmp / "clickhouse"
        with tarfile.open(tgz_path, "r:gz") as tf:
            member = next(
                (m for m in tf.getmembers() if m.name.endswith("/usr/bin/clickhouse")),
                None,
            )
            if member is None:
                raise ValueError(f"usr/bin/clickhouse not found in {tgz_path.name}")
            src = tf.extractfile(member)
            if src is None:
                raise ValueError(f"unable to extract {member.name}")
            with extracted_bin.open("wb") as out:
                shutil.copyfileobj(src, out, length=1 << 20)

        extracted_bin.chmod(0o755)
        tmp_dest = dest.with_name(f".{dest.name}.tmp.{os.getpid()}")
        shutil.copy2(extracted_bin, tmp_dest)
        tmp_dest.chmod(
            stat.S_IRUSR
            | stat.S_IWUSR
            | stat.S_IXUSR
            | stat.S_IRGRP
            | stat.S_IXGRP
            | stat.S_IROTH
            | stat.S_IXOTH
        )
        tmp_dest.replace(dest)

    ver_str = clickhouse_version(dest)
    return {
        "binary": str(dest.resolve()),
        "release_tag": tag,
        "arch": arch,
        "sha512": expected_sha,
        "version": ver_str,
    }


def clickhouse_version(binary: Path) -> str:
    res = subprocess.run(
        [str(binary), "local", "--version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )
    if res.returncode != 0:
        raise RuntimeError(
            f"clickhouse --version failed (rc={res.returncode}): {res.stderr.strip()}"
        )
    return res.stdout.strip()


def cache_pages(files: list[Path]) -> dict[str, int] | None:
    """Sample Linux file page-cache residency without faulting in the data."""
    if platform.system() != "Linux":
        return None
    mincore = ctypes.CDLL(None, use_errno=True).mincore
    resident = total = 0
    for path in files:
        with path.open("rb") as file, mmap.mmap(
            file.fileno(), 0, access=mmap.ACCESS_COPY
        ) as mem:
            pages = (len(mem) + mmap.PAGESIZE - 1) // mmap.PAGESIZE
            status = (ctypes.c_ubyte * pages)()
            address = ctypes.addressof(ctypes.c_char.from_buffer(mem))
            if (
                mincore(ctypes.c_void_p(address), ctypes.c_size_t(len(mem)), status)
                != 0
            ):
                raise OSError(ctypes.get_errno(), "mincore failed", str(path))
            resident += sum(p & 1 for p in status)
            total += pages
    return {"resident": resident, "total": total}


def warm_files(files: list[Path]):
    # Read each file twice so Linux promotes its pages to the active file LRU
    # list and avoids reclaiming unread Parquet columns during heavy hash aggs.
    for path in files:
        for _ in range(2):
            with path.open("rb") as file:
                while file.read(1 << 20):
                    pass


def require_warm_cache(files: list[Path]) -> dict[str, int]:
    pages = cache_pages(files)
    if pages is None:
        raise ValueError("warm cache verification requires Linux mincore")
    if pages["resident"] != pages["total"]:
        raise ValueError(f"cannot verify warm file page cache: {pages}")
    return pages


def evict_files(files: list[Path]):
    """Drop clean file pages from the Linux page cache without root privileges."""
    if platform.system() != "Linux" or not hasattr(os, "posix_fadvise"):
        raise ValueError("cold cache eviction requires Linux posix_fadvise")
    for path in files:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def require_cold_cache(files: list[Path]) -> dict[str, int]:
    pages = cache_pages(files)
    if pages is None:
        raise ValueError("cold cache verification requires Linux mincore")
    if pages["resident"] != 0:
        raise ValueError(f"cannot verify cold file page cache: {pages}")
    return pages


class DataFusionRunner:
    def __init__(
        self,
        binary: Path,
        data_path: Path,
        threads: int,
        timeout: int,
        mode: str,
        profile: str,
    ):
        self.binary = binary.resolve(strict=True)
        self.data_path = data_path.resolve(strict=True)
        self.threads = threads
        self.timeout = timeout
        self.mode = mode
        self.profile = profile

    def _run(self, operation: str, spec: QuerySpec) -> bytes:
        result = subprocess.run(
            [
                str(self.binary),
                operation,
                self.mode,
                self.profile,
                str(self.threads),
                str(self.data_path),
                spec.sql,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=self.timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"clickbench_runner {operation} ({self.mode}/{self.profile}) failed on "
                f"{spec.name} (rc={result.returncode}): "
                f"{result.stderr.decode(errors='replace').strip()}"
            )
        return result.stdout

    def query(self, spec: QuerySpec) -> pa.Table:
        raw = self._run("query", spec)
        return pa_ipc.open_stream(raw).read_all()

    def measure(self, spec: QuerySpec) -> dict:
        return json.loads(self._run("measure", spec))

    def plan(self, spec: QuerySpec) -> dict:
        return json.loads(self._run("plan", spec))


class DuckDbRunner:
    def __init__(self, data_path: Path, threads: int, timeout: int):
        self.data_path = data_path.resolve(strict=True)
        self.threads = threads
        self.timeout = timeout
        self.script = Path(__file__).resolve()

    def _run(self, operation: str, spec: QuerySpec) -> bytes:
        sql = adapt_sql_for_duckdb(spec.name, spec.sql)
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                str(self.script),
                "_duckdb-exec",
                operation,
                str(self.threads),
                str(self.data_path),
                sql,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=self.timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"duckdb {operation} failed on {spec.name} "
                f"(rc={result.returncode}): "
                f"{result.stderr.decode(errors='replace').strip()}"
            )
        return result.stdout

    def query(self, spec: QuerySpec) -> pa.Table:
        raw = self._run("query", spec)
        return pa_ipc.open_stream(raw).read_all()

    def measure(self, spec: QuerySpec) -> dict:
        return json.loads(self._run("measure", spec))


def _parquet_glob(data_path: Path) -> str:
    if data_path.is_file():
        return data_path.as_posix().replace("'", "''")
    return (data_path / "*.parquet").as_posix().replace("'", "''")


def _duckdb_exec(operation: str, threads: int, data_path: Path, sql: str):
    con = duckdb.connect(":memory:")
    con.execute(f"SET threads = {threads}")
    con.execute("SET TimeZone = 'UTC'")
    # Load Parquet extension and initialize Arrow bridge before starting the clock so
    # one-time extension loading does not skew query timing vs statically linked engines.
    con.execute("LOAD parquet")
    for _ in con.sql("SELECT 1").to_arrow_reader(batch_size=BATCH_SIZE):
        pass

    glob_pattern = _parquet_glob(data_path)
    view_ddl = (
        "CREATE VIEW hits AS "
        'SELECT * EXCLUDE ("EventDate"), '
        '(DATE \'1970-01-01\' + CAST("EventDate" AS INTEGER)) AS "EventDate" '
        f"FROM read_parquet('{glob_pattern}', binary_as_string=true)"
    )

    if operation == "query":
        con.execute(view_ddl)
        tbl = con.sql(sql).to_arrow_table()
        with pa_ipc.new_stream(sys.stdout.buffer, tbl.schema) as writer:
            writer.write_table(tbl)
        return

    if operation == "measure":
        start = time.perf_counter()
        cpu_start = time.process_time()
        con.execute(view_ddl)
        register_seconds = time.perf_counter() - start
        exec_start = time.perf_counter()
        reader = con.sql(sql).to_arrow_reader(batch_size=BATCH_SIZE)
        output_rows = 0
        output_batches = 0
        first_output_seconds = None
        for batch in reader:
            if batch.num_rows:
                output_rows += batch.num_rows
                output_batches += 1
                if output_batches == 1:
                    first_output_seconds = time.perf_counter() - start
        exec_seconds = time.perf_counter() - exec_start
        seconds = time.perf_counter() - start
        exec_cpu_seconds = time.process_time() - cpu_start
        print(
            json.dumps(
                {
                    "duckdb_version": duckdb.__version__,
                    "output_rows": output_rows,
                    "output_batches": output_batches,
                    "register_seconds": register_seconds,
                    "plan_seconds": None,
                    "exec_seconds": exec_seconds,
                    "exec_cpu_seconds": exec_cpu_seconds,
                    "first_output_seconds": first_output_seconds,
                    "seconds": seconds,
                }
            )
        )
        return

    raise ValueError(f"unknown _duckdb-exec operation: {operation}")


def _parse_clickhouse_times(stderr_text: str) -> list[float]:
    times: list[float] = []
    for line in stderr_text.splitlines():
        stripped = line.strip()
        if re.fullmatch(r"\d+\.\d+", stripped):
            times.append(float(stripped))
    return times


class ClickHouseRunner:
    def __init__(self, binary: Path, data_path: Path, threads: int, timeout: int):
        self.binary = binary.resolve(strict=True)
        self.data_path = data_path.resolve(strict=True)
        self.threads = threads
        self.timeout = timeout
        self.version = clickhouse_version(self.binary)

    def _view_ddl(self) -> str:
        glob_pattern = _parquet_glob(self.data_path)
        return (
            "CREATE VIEW hits AS "
            "SELECT * EXCEPT (EventDate), toDate(EventDate) AS EventDate "
            f"FROM file('{glob_pattern}', Parquet)"
        )

    def _exec(
        self, spec: QuerySpec, fmt: str, timed: bool
    ) -> subprocess.CompletedProcess:
        adapted = adapt_sql_for_clickhouse(spec.name, spec.sql)
        full_sql = f"{self._view_ddl()};\n{adapted}"
        cmd = [
            str(self.binary),
            "local",
            "--tmp",
            f"--max_threads={self.threads}",
            f"--format={fmt}",
        ]
        if timed:
            cmd.append("--time")
        cmd.extend(["-q", full_sql])
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=self.timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"clickhouse local failed on {spec.name} (rc={result.returncode}): "
                f"{result.stderr.decode(errors='replace').strip()}"
            )
        return result

    def query(self, spec: QuerySpec) -> pa.Table:
        res = self._exec(spec, fmt="ArrowStream", timed=False)
        if not res.stdout:
            return pa.table({})
        return pa_ipc.open_stream(res.stdout).read_all()

    def measure(self, spec: QuerySpec) -> dict:
        wall_start = time.perf_counter()
        res = self._exec(spec, fmt="ArrowStream", timed=True)
        wall_seconds = time.perf_counter() - wall_start
        stderr_text = res.stderr.decode(errors="replace")
        stmt_times = _parse_clickhouse_times(stderr_text)
        if len(stmt_times) != 2:
            raise RuntimeError(
                f"expected 2 statement timings from clickhouse --time on {spec.name}, "
                f"got {stmt_times} (stderr={stderr_text!r})"
            )
        register_seconds, exec_seconds = stmt_times
        if res.stdout:
            reader = pa_ipc.open_stream(res.stdout)
            output_rows = 0
            output_batches = 0
            for batch in reader:
                if batch.num_rows:
                    output_rows += batch.num_rows
                    output_batches += 1
        else:
            output_rows = 0
            output_batches = 0
        return {
            "clickhouse_version": self.version,
            "output_rows": output_rows,
            "output_batches": output_batches,
            "register_seconds": register_seconds,
            "plan_seconds": None,
            "exec_seconds": exec_seconds,
            "exec_cpu_seconds": None,
            "first_output_seconds": None,
            "seconds": register_seconds + exec_seconds,
            "process_wall_seconds": wall_seconds,
        }


def _normalize_value(val):
    if val is None:
        return None
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="surrogateescape")
    if isinstance(val, datetime.datetime):
        if val.tzinfo is not None:
            return val.astimezone(datetime.timezone.utc).replace(tzinfo=None)
        return val
    return val


def _values_equal(
    expected, actual, rel_tol: float = 1e-9, abs_tol: float = 1e-9
) -> bool:
    exp = _normalize_value(expected)
    act = _normalize_value(actual)
    if exp is None or act is None:
        return exp is act
    if isinstance(exp, Decimal) and isinstance(act, Decimal):
        return exp == act
    if isinstance(exp, Decimal) and isinstance(act, int):
        return exp == Decimal(act)
    if isinstance(exp, int) and isinstance(act, Decimal):
        return Decimal(exp) == act
    if isinstance(exp, (float, Decimal)) and isinstance(act, (float, Decimal)):
        return math.isclose(float(exp), float(act), rel_tol=rel_tol, abs_tol=abs_tol)
    if isinstance(exp, (int, float)) and isinstance(act, (int, float)):
        if isinstance(exp, int) and isinstance(act, int):
            return exp == act
        return math.isclose(float(exp), float(act), rel_tol=rel_tol, abs_tol=abs_tol)
    return exp == act


def _rows_equal(exp_row: tuple, act_row: tuple) -> bool:
    return all(_values_equal(e, a) for e, a in zip(exp_row, act_row))


def _sort_key(row: tuple):
    key = []
    for raw in row:
        v = _normalize_value(raw)
        if v is None:
            key.append((0, 0))
        elif isinstance(v, bool):
            key.append((1, int(v)))
        elif isinstance(v, int):
            key.append((1, Decimal(v)))
        elif isinstance(v, Decimal):
            key.append((1, v))
        elif isinstance(v, float):
            if math.isnan(v):
                key.append((2, 0))
            else:
                # Round floats slightly for deterministic tie-group bucketing; exact
                # tolerance check runs after sorting.
                key.append((1, round(v, 7)))
        else:
            key.append((1, v))
    return tuple(key)


def _extract_key(row: tuple, key_cols: tuple[int, ...]) -> tuple:
    return tuple(row[i] for i in key_cols)


def _table_rows(tbl: pa.Table) -> list[tuple]:
    if tbl.num_rows == 0:
        return []
    cols = [col.to_pylist() for col in tbl.columns]
    return list(zip(*cols))


_TRAILING_ORDER_LIMIT_RE = re.compile(
    r"(?:\s+ORDER\s+BY\s+.+?)?\s+LIMIT\s+\d+(?:\s+OFFSET\s+\d+)?\s*$",
    re.IGNORECASE | re.DOTALL,
)


def strip_order_limit(sql: str) -> str:
    """Remove trailing ORDER BY / LIMIT / OFFSET to query the full oracle relation."""
    stripped = _TRAILING_ORDER_LIMIT_RE.sub("", sql.strip().rstrip(";")).strip()
    if not stripped:
        raise ValueError(f"failed to strip ORDER BY/LIMIT from SQL: {sql!r}")
    return stripped


def _arrow_column_for_duckdb_key(col: pa.ChunkedArray | pa.Array) -> pa.Array:
    typ = col.type
    if (
        pa.types.is_binary(typ)
        or pa.types.is_large_binary(typ)
        or pa.types.is_binary_view(typ)
    ):
        return pa.array(
            [_normalize_value(v) for v in col.to_pylist()], type=pa.string()
        )
    if pa.types.is_timestamp(typ) and typ.tz is not None:
        return pa.array(
            [_normalize_value(v) for v in col.to_pylist()],
            type=pa.timestamp(typ.unit),
        )
    if isinstance(col, pa.ChunkedArray):
        return col.combine_chunks()
    return col


class DuckDbOracle:
    """In-memory DuckDB oracle for verifying tied LIMIT/OFFSET boundary rows and q17/q24."""

    def __init__(self, data_path: Path, threads: int):
        self.data_path = data_path.resolve(strict=True)
        self.threads = threads
        self._con: duckdb.DuckDBPyConnection | None = None
        self._q24_cache: dict[tuple[str, int | None], pa.Table] = {}

    def __enter__(self) -> "DuckDbOracle":
        con = duckdb.connect(":memory:")
        con.execute(f"SET threads = {self.threads}")
        con.execute("SET TimeZone = 'UTC'")
        glob_pattern = _parquet_glob(self.data_path)
        con.execute(
            "CREATE VIEW hits AS "
            'SELECT * EXCLUDE ("EventDate"), '
            '(DATE \'1970-01-01\' + CAST("EventDate" AS INTEGER)) AS "EventDate" '
            f"FROM read_parquet('{glob_pattern}', binary_as_string=true)"
        )
        self._con = con
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._con is not None:
            self._con.close()
            self._con = None

    def _require_con(self) -> duckdb.DuckDBPyConnection:
        if self._con is None:
            raise RuntimeError("DuckDbOracle must be used inside a context manager")
        return self._con

    def lookup_candidates(
        self, spec: QuerySpec, actual_subset: pa.Table, key_cols: tuple[int, ...]
    ) -> pa.Table:
        """Query the full un-limited oracle relation filtered to candidate key tuples."""
        con = self._require_con()
        adapted = adapt_sql_for_duckdb(spec.name, spec.sql)
        base_sql = strip_order_limit(adapted)
        ncols = actual_subset.num_columns
        col_aliases = ", ".join(f"c{i}" for i in range(ncols))

        key_arrays = {
            f"k{idx}": _arrow_column_for_duckdb_key(actual_subset.column(col_idx))
            for idx, col_idx in enumerate(key_cols)
        }
        keys_tbl = pa.table(key_arrays)
        join_cond = " AND ".join(
            f"_o.c{col_idx} IS NOT DISTINCT FROM _keys.k{idx}"
            for idx, col_idx in enumerate(key_cols)
        )
        sql = (
            f"WITH _o({col_aliases}) AS (\n{base_sql}\n)\n"
            f"SELECT _o.* FROM _o SEMI JOIN _keys ON {join_cond}"
        )
        con.register("_keys", keys_tbl)
        try:
            return con.sql(sql).to_arrow_table()
        finally:
            con.unregister("_keys")

    def q24_oracle(self, spec: QuerySpec) -> pa.Table:
        """Return all (EventTime, SearchPhrase) rows with EventTime <= Nth smallest."""
        cache_key = (spec.name, spec.limit)
        if cache_key in self._q24_cache:
            return self._q24_cache[cache_key]
        con = self._require_con()
        limit = spec.limit if spec.limit is not None else 10
        sql = (
            "WITH topk AS (\n"
            '  SELECT "EventTime" FROM hits WHERE "SearchPhrase" <> \'\' '
            f'ORDER BY "EventTime" LIMIT {limit}\n'
            "),\n"
            'cutoff AS (SELECT MAX("EventTime") AS max_t FROM topk)\n'
            'SELECT "EventTime", "SearchPhrase"\n'
            "FROM hits, cutoff\n"
            'WHERE "SearchPhrase" <> \'\' AND "EventTime" <= cutoff.max_t\n'
            'ORDER BY "EventTime"'
        )
        tbl = con.sql(sql).to_arrow_table()
        self._q24_cache[cache_key] = tbl
        return tbl


def _normalized_key_tuple(row: tuple, key_cols: tuple[int, ...]) -> tuple:
    parts = []
    for c in key_cols:
        v = _normalize_value(row[c])
        if isinstance(v, Decimal) and v == int(v):
            v = int(v)
        parts.append(v)
    return tuple(parts)


def _check_unique_keys(
    rows: list[tuple], key_cols: tuple[int, ...], case_name: str, arm_name: str
):
    seen: set[tuple] = set()
    for idx, row in enumerate(rows):
        key = _normalized_key_tuple(row, key_cols)
        if key in seen:
            raise ValueError(
                f"{arm_name}/{case_name}: duplicate group key {key!r} at row {idx}"
            )
        seen.add(key)


def _is_multiset_equal(exp_rows: list[tuple], act_rows: list[tuple]) -> bool:
    if len(exp_rows) != len(act_rows):
        return False
    exp_sorted = sorted(exp_rows, key=_sort_key)
    act_sorted = sorted(act_rows, key=_sort_key)
    return all(_rows_equal(er, ar) for er, ar in zip(exp_sorted, act_sorted))


def _check_multiset_equal(
    exp_rows: list[tuple],
    act_rows: list[tuple],
    case_name: str,
    arm_name: str,
    context: str = "",
):
    if len(exp_rows) != len(act_rows):
        raise ValueError(
            f"{arm_name}/{case_name}{context}: group size mismatch: "
            f"expected {len(exp_rows)}, got {len(act_rows)}"
        )
    exp_sorted = sorted(exp_rows, key=_sort_key)
    act_sorted = sorted(act_rows, key=_sort_key)
    for idx, (er, ar) in enumerate(zip(exp_sorted, act_sorted)):
        if not _rows_equal(er, ar):
            raise ValueError(
                f"{arm_name}/{case_name}{context}: row mismatch at sorted index {idx}: "
                f"expected {er!r}, got {ar!r}"
            )


def _resolve_oracle_candidates(
    oracle_lookup: (
        Callable[[QuerySpec, pa.Table, tuple[int, ...]], pa.Table] | pa.Table | None
    ),
    spec: QuerySpec,
    actual_subset: pa.Table,
    key_cols: tuple[int, ...],
    arm_name: str,
) -> list[tuple]:
    if oracle_lookup is None:
        raise ValueError(
            f"{arm_name}/{spec.name}: oracle_lookup is required to verify boundary/unordered rows"
        )
    tbl = (
        oracle_lookup
        if isinstance(oracle_lookup, pa.Table)
        else oracle_lookup(spec, actual_subset, key_cols)
    )
    return _table_rows(tbl)


def _verify_group_rows_against_candidates(
    actual_slice: list[tuple],
    candidate_rows: list[tuple],
    group_key_cols: tuple[int, ...],
    case_name: str,
    arm_name: str,
    context: str = "",
):
    by_key: dict[tuple, tuple] = {}
    for er in candidate_rows:
        by_key[_normalized_key_tuple(er, group_key_cols)] = er
    for idx, ar in enumerate(actual_slice):
        key = _normalized_key_tuple(ar, group_key_cols)
        if key not in by_key:
            raise ValueError(
                f"{arm_name}/{case_name}{context}: group key {key!r} at row {idx} "
                "not found in oracle full GROUP BY result"
            )
        er = by_key[key]
        if not _rows_equal(er, ar):
            raise ValueError(
                f"{arm_name}/{case_name}{context}: aggregate mismatch for group {key!r}: "
                f"expected {er!r}, got {ar!r}"
            )


def _verify_multiset_subset(
    actual_slice: list[tuple],
    candidate_rows: list[tuple],
    case_name: str,
    arm_name: str,
    context: str = "",
):
    used = [False] * len(candidate_rows)
    for idx, ar in enumerate(actual_slice):
        matched = False
        for cand_idx, er in enumerate(candidate_rows):
            if not used[cand_idx] and _rows_equal(er, ar):
                used[cand_idx] = True
                matched = True
                break
        if not matched:
            raise ValueError(
                f"{arm_name}/{case_name}{context}: boundary row {idx} ({ar!r}) "
                "not found in oracle candidate pool (or exceeds multiplicity)"
            )


def _verify_q24_against_oracle(
    actual_rows: list[tuple],
    q24_oracle_rows: list[tuple],
    case_name: str,
    arm_name: str,
):
    n = len(actual_rows)
    if len(q24_oracle_rows) < n:
        raise ValueError(
            f"{arm_name}/{case_name}: oracle q24 pool has only {len(q24_oracle_rows)} rows, "
            f"fewer than actual {n}"
        )
    sorted_oracle = sorted(q24_oracle_rows, key=lambda r: _normalize_value(r[0]))
    cutoff_t = _normalize_value(sorted_oracle[n - 1][0])

    # Every EventTime strictly below cutoff_t must appear in ascending EventTime
    # order with exact multiset equality within each equal-EventTime group.
    strict_rows = [
        (_normalize_value(t), _normalize_value(p))
        for t, p in sorted_oracle
        if _normalize_value(t) < cutoff_t
    ]
    pos = 0
    idx = 0
    while idx < len(strict_rows):
        t_val = strict_rows[idx][0]
        end = idx + 1
        while end < len(strict_rows) and strict_rows[end][0] == t_val:
            end += 1
        expected_phrases = Counter(p for _, p in strict_rows[idx:end])
        group_len = end - idx
        actual_phrases = Counter(
            _normalize_value(r[0]) for r in actual_rows[pos : pos + group_len]
        )
        if actual_phrases != expected_phrases:
            raise ValueError(
                f"{arm_name}/{case_name}: strict EventTime={t_val} prefix mismatch "
                f"at rows [{pos}..{pos + group_len}]: "
                f"expected {dict(expected_phrases)}, got {dict(actual_phrases)}"
            )
        pos += group_len
        idx = end

    # Remaining rows must come from rows with EventTime == cutoff_t (respecting multiplicity).
    boundary_pool = Counter(
        _normalize_value(p) for t, p in sorted_oracle if _normalize_value(t) == cutoff_t
    )
    tail_counts = Counter(_normalize_value(r[0]) for r in actual_rows[pos:])
    for phrase, cnt in tail_counts.items():
        allowed = boundary_pool.get(phrase, 0)
        if cnt > allowed:
            raise ValueError(
                f"{arm_name}/{case_name}: phrase {phrase!r} appears {cnt} time(s) "
                f"at cutoff EventTime={cutoff_t}, exceeding oracle multiplicity {allowed}"
            )


def compare_tables(
    oracle: pa.Table,
    actual: pa.Table,
    spec: QuerySpec,
    arm_name: str,
    *,
    oracle_lookup: (
        Callable[[QuerySpec, pa.Table, tuple[int, ...]], pa.Table] | pa.Table | None
    ) = None,
    oracle_q24: Callable[[QuerySpec], pa.Table] | pa.Table | None = None,
):
    """Verify lossless Arrow result parity against the oracle with strict SQL tie checks."""
    # ClickHouse ArrowStream writes an empty stream when a query returns 0 rows.
    if oracle.num_rows == 0 and actual.num_rows == 0:
        if actual.num_columns != 0 and oracle.num_columns != actual.num_columns:
            raise ValueError(
                f"{arm_name}/{spec.name}: column count mismatch on empty result: "
                f"expected {oracle.num_columns}, got {actual.num_columns}"
            )
        return

    if oracle.num_columns != actual.num_columns:
        raise ValueError(
            f"{arm_name}/{spec.name}: column count mismatch: "
            f"expected {oracle.num_columns} ({oracle.schema.names}), "
            f"got {actual.num_columns} ({actual.schema.names})"
        )
    if oracle.num_rows != actual.num_rows:
        raise ValueError(
            f"{arm_name}/{spec.name}: row count mismatch: "
            f"expected {oracle.num_rows}, got {actual.num_rows}"
        )

    oracle_rows = _table_rows(oracle)
    actual_rows = _table_rows(actual)

    if spec.order_mode == "exact":
        for row_idx, (er, ar) in enumerate(zip(oracle_rows, actual_rows)):
            for col_idx, (ev, av) in enumerate(zip(er, ar)):
                if not _values_equal(ev, av):
                    raise ValueError(
                        f"{arm_name}/{spec.name}: mismatch at row {row_idx}, "
                        f"col {col_idx} ({oracle.schema.names[col_idx]!r}): "
                        f"expected {ev!r}, got {av!r}"
                    )
        return

    if spec.order_mode == "unordered":
        _check_multiset_equal(oracle_rows, actual_rows, spec.name, arm_name)
        return

    if spec.order_mode == "limit_unordered":
        # q17: GROUP BY ... LIMIT 10 with no ORDER BY.
        # Every returned group must be unique and exist in the oracle's full GROUP BY
        # with identical aggregate values.
        if not spec.group_key_cols:
            raise ValueError(
                f"{arm_name}/{spec.name}: limit_unordered requires group_key_cols"
            )
        _check_unique_keys(actual_rows, spec.group_key_cols, spec.name, arm_name)
        if _is_multiset_equal(oracle_rows, actual_rows):
            return
        candidates = _resolve_oracle_candidates(
            oracle_lookup, spec, actual, spec.group_key_cols, arm_name
        )
        _verify_group_rows_against_candidates(
            actual_rows, candidates, spec.group_key_cols, spec.name, arm_name
        )
        return

    if spec.order_mode == "topn_hidden_key":
        # q24: SELECT "SearchPhrase" ... ORDER BY "EventTime" LIMIT 10.
        # Verify against oracle rows including EventTime so every row below the Nth
        # EventTime appears in order and remaining rows come from the Nth EventTime.
        if oracle_q24 is not None:
            q24_tbl = (
                oracle_q24 if isinstance(oracle_q24, pa.Table) else oracle_q24(spec)
            )
            _verify_q24_against_oracle(
                actual_rows, _table_rows(q24_tbl), spec.name, arm_name
            )
            return
        # Without oracle_q24, fall back strictly to exact row-by-row equality.
        for row_idx, (er, ar) in enumerate(zip(oracle_rows, actual_rows)):
            if not _rows_equal(er, ar):
                raise ValueError(
                    f"{arm_name}/{spec.name}: mismatch at row {row_idx} "
                    f"(provide oracle_q24 for EventTime-tie verification): "
                    f"expected {er!r}, got {ar!r}"
                )
        return

    if spec.order_mode in ("ordered_ties", "topn"):
        key_cols = spec.sort_key_cols
        # 1. Verify exact sequence of ORDER BY sort-key values at every row position.
        for idx, (er, ar) in enumerate(zip(oracle_rows, actual_rows)):
            ek = _extract_key(er, key_cols)
            ak = _extract_key(ar, key_cols)
            if not _rows_equal(ek, ak):
                raise ValueError(
                    f"{arm_name}/{spec.name}: sort key mismatch at row {idx}: "
                    f"expected {ek!r}, got {ak!r}"
                )

        # 2. For GROUP BY queries, ensure no duplicate group keys appear anywhere.
        if spec.group_key_cols:
            _check_unique_keys(actual_rows, spec.group_key_cols, spec.name, arm_name)

        # 3. Partition rows into contiguous runs of equal sort keys.
        runs: list[tuple[int, int]] = []
        i = 0
        n = len(oracle_rows)
        while i < n:
            j = i + 1
            base_key = _extract_key(oracle_rows[i], key_cols)
            while j < n and _rows_equal(
                base_key, _extract_key(oracle_rows[j], key_cols)
            ):
                j += 1
            runs.append((i, j))
            i = j

        # 4. Every non-boundary run must match oracle_rows[lo:hi] as an exact multiset.
        # A final run is only a LIMIT boundary when the result was actually cut by LIMIT
        # (n == spec.limit). Any boundary run that differs from oracle_rows[lo:hi] is
        # verified row-by-row against the oracle's full un-limited relation.
        for lo, hi in runs:
            is_end_boundary = (
                spec.order_mode == "topn"
                and hi == n
                and spec.limit is not None
                and n == spec.limit
            )
            is_start_boundary = (
                spec.order_mode == "topn" and spec.has_offset and lo == 0
            )
            ctx = f"[rows {lo}..{hi}]"
            if not (is_end_boundary or is_start_boundary):
                _check_multiset_equal(
                    oracle_rows[lo:hi],
                    actual_rows[lo:hi],
                    spec.name,
                    arm_name,
                    context=ctx,
                )
                continue
            if _is_multiset_equal(oracle_rows[lo:hi], actual_rows[lo:hi]):
                continue
            actual_slice_tbl = actual.slice(lo, hi - lo)
            if spec.group_key_cols:
                candidates = _resolve_oracle_candidates(
                    oracle_lookup,
                    spec,
                    actual_slice_tbl,
                    spec.group_key_cols,
                    arm_name,
                )
                _verify_group_rows_against_candidates(
                    actual_rows[lo:hi],
                    candidates,
                    spec.group_key_cols,
                    spec.name,
                    arm_name,
                    context=ctx,
                )
            else:
                candidates = _resolve_oracle_candidates(
                    oracle_lookup,
                    spec,
                    actual_slice_tbl,
                    key_cols,
                    arm_name,
                )
                _verify_multiset_subset(
                    actual_rows[lo:hi],
                    candidates,
                    spec.name,
                    arm_name,
                    context=ctx,
                )
        return

    raise ValueError(f"unknown order_mode: {spec.order_mode}")


def categorize_morsel_rejection(error: str) -> str:
    """Group a morsel_mode=required planning rejection into a roadmap category."""
    low = error.lower()
    if "distinct" in low:
        return "DISTINCT / COUNT(DISTINCT) aggregation"
    if "offset" in low or "fetch" in low or "limit" in low:
        return "LIMIT / OFFSET operator"
    if "window" in low:
        return "Window operator"
    if "aggregate" in low or "accumulator" in low or "group" in low:
        return "Unsupported aggregate expression / mode"
    if "sort" in low or "topn" in low:
        return "Unsupported Sort / TopN"
    if "projection" in low or "expression" in low or "cast" in low:
        return "Unsupported expression / Projection"
    if "filter" in low:
        return "Unsupported Filter"
    if "parquet" in low or "datasource" in low or "scan" in low:
        return "Unsupported Parquet / DataSource scan"
    if "join" in low:
        return "Unsupported Join"
    return "Other unsupported physical operator"


def group_morsel_rejections(rejections: dict[str, str]) -> list[dict]:
    """Return deterministic grouped rejection summaries sorted by descending count."""
    buckets: dict[str, dict] = {}
    for qname, err in rejections.items():
        cat = categorize_morsel_rejection(err)
        entry = buckets.setdefault(
            cat,
            {
                "category": cat,
                "queries": [],
                "representative_error": err,
            },
        )
        entry["queries"].append(qname)
    groups = list(buckets.values())
    for g in groups:
        g["count"] = len(g["queries"])
    groups.sort(key=lambda g: (-g["count"], g["category"]))
    return groups


def validate_plan(evidence: dict, arm: Arm, spec: QuerySpec):
    if arm.engine != "datafusion":
        return
    plan = evidence["plan"]
    if arm.mode == "off":
        # Stock DataFusion may fold metadata-only queries (e.g., q0 COUNT(*), q6 MIN/MAX)
        # into PlaceholderRowExec via AggregateStatistics.
        has_stock_scan = bool(re.search(r"DataSourceExec: .*file_type=parquet", plan))
        has_stats_fold = "PlaceholderRowExec" in plan
        if "MorselExec:" in plan or not (has_stock_scan or has_stats_fold):
            raise ValueError(
                f"{arm.name}/{spec.name}: expected a stock Parquet plan, got:\n{plan}"
            )
        return
    if not plan.startswith("MorselExec:"):
        raise ValueError(
            f"{arm.name}/{spec.name}: expected MorselExec root under morsel_mode=required, got:\n{plan}"
        )


def probe_morsel_matrix(
    data_path: Path,
    runner_bin: Path,
    threads: int,
    timeout: int,
    query_names: list[str],
    profile: str = "pushdown",
) -> dict:
    """Probe morsel_mode=required planning admission across the selected queries."""
    morsel = DataFusionRunner(
        runner_bin, data_path, threads, timeout, mode="required", profile=profile
    )
    admitted: list[str] = []
    rejections: dict[str, str] = {}
    plans: dict[str, str] = {}
    revision = None
    for qname in query_names:
        spec = QUERIES[qname]
        res = morsel.plan(spec)
        revision = res.get("build_revision")
        if res["admitted"]:
            admitted.append(qname)
            plans[qname] = res["plan"]
        else:
            rejections[qname] = res["error"] or "rejected without error message"
    return {
        "build_revision": revision,
        "profile": profile,
        "total_queries": len(query_names),
        "admitted_count": len(admitted),
        "rejected_count": len(rejections),
        "admitted": admitted,
        "rejections": rejections,
        "grouped_rejections": group_morsel_rejections(rejections),
        "plans": plans,
    }


def _build_runners(
    data_path: Path,
    runner_bin: Path | None,
    clickhouse_bin: Path | None,
    threads: int,
    timeout: int,
    arm_names: list[str],
) -> dict[str, DataFusionRunner | DuckDbRunner | ClickHouseRunner]:
    runners: dict[str, DataFusionRunner | DuckDbRunner | ClickHouseRunner] = {}
    for arm_name in arm_names:
        arm = ARMS[arm_name]
        if arm.engine == "datafusion":
            if runner_bin is None:
                raise ValueError(
                    f"--runner is required for DataFusion arm '{arm_name}'"
                )
            runners[arm_name] = DataFusionRunner(
                runner_bin, data_path, threads, timeout, arm.mode, arm.profile
            )
        elif arm.engine == "duckdb":
            runners[arm_name] = DuckDbRunner(data_path, threads, timeout)
        elif arm.engine == "clickhouse":
            if clickhouse_bin is None:
                raise ValueError(
                    f"--clickhouse is required for ClickHouse arm '{arm_name}'"
                )
            runners[arm_name] = ClickHouseRunner(
                clickhouse_bin, data_path, threads, timeout
            )
        else:
            raise ValueError(f"unknown engine: {arm.engine}")
    return runners


def verify_suite(
    data_path: Path,
    runner_bin: Path,
    clickhouse_bin: Path,
    threads: int,
    timeout: int,
    query_names: list[str],
    arm_names: list[str],
) -> dict:
    """Verify 4-way Arrow result parity and Morsel admission/rejection checks."""
    runners = _build_runners(
        data_path, runner_bin, clickhouse_bin, threads, timeout, arm_names
    )
    # Oracle is duckdb_default if present, else the first configured arm.
    oracle_arm = (
        "duckdb_default" if "duckdb_default" in runners else next(iter(runners.keys()))
    )
    morsel_arms = [a for a in arm_names if ARMS[a].mode == "required"]
    matrix = probe_morsel_matrix(
        data_path,
        runner_bin,
        threads,
        timeout,
        query_names,
        profile=ARMS[morsel_arms[0]].profile if morsel_arms else "pushdown",
    )
    admitted_set = set(matrix["admitted"])

    verified: dict[str, dict] = {}
    with DuckDbOracle(data_path, threads) as oracle_db:
        for qname in query_names:
            spec = QUERIES[qname]
            oracle_tbl = runners[oracle_arm].query(spec)
            checked_arms: list[str] = [oracle_arm]
            for arm_name in arm_names:
                if arm_name == oracle_arm:
                    continue
                arm = ARMS[arm_name]
                if arm.mode == "required" and qname not in admitted_set:
                    # Confirm rejection under this specific morsel arm's profile as well.
                    plan_res = runners[arm_name].plan(spec)
                    if plan_res["admitted"]:
                        raise ValueError(
                            f"{arm_name}/{qname}: unexpectedly admitted on verify"
                        )
                    continue
                actual_tbl = runners[arm_name].query(spec)
                compare_tables(
                    oracle_tbl,
                    actual_tbl,
                    spec,
                    arm_name,
                    oracle_lookup=oracle_db.lookup_candidates,
                    oracle_q24=oracle_db.q24_oracle,
                )
                checked_arms.append(arm_name)
            verified[qname] = {
                "rows": oracle_tbl.num_rows,
                "columns": oracle_tbl.num_columns,
                "oracle": oracle_arm,
                "checked_arms": checked_arms,
                "morsel_admitted": qname in admitted_set,
                "morsel_rejection": matrix["rejections"].get(qname),
            }
    return {
        "dataset": dataset_summary(data_path),
        "threads": threads,
        "arms": arm_names,
        "morsel_admitted_count": matrix["admitted_count"],
        "morsel_rejected_count": matrix["rejected_count"],
        "grouped_rejections": matrix["grouped_rejections"],
        "queries": verified,
    }


def run_bench(args):
    data_path = args.path.resolve(strict=True)
    files = dataset_files(data_path)
    summary = dataset_summary(data_path)
    query_names = resolve_queries(args.queries)
    arm_names = resolve_arms(args.arms, DEFAULT_BENCH_ARMS)

    if args.cache in ("warm", "cold") and platform.system() != "Linux":
        raise ValueError(
            f"--cache {args.cache} requires Linux mincore page-cache verification"
        )

    runners = _build_runners(
        data_path,
        args.runner,
        args.clickhouse,
        args.threads,
        args.timeout,
        arm_names,
    )

    # Probe Morsel admission across all selected queries before warming/timing.
    morsel_arms = [a for a in arm_names if ARMS[a].mode == "required"]
    matrix = None
    if morsel_arms and args.runner is not None:
        matrix = probe_morsel_matrix(
            data_path,
            args.runner,
            args.threads,
            args.timeout,
            query_names,
            profile=ARMS[morsel_arms[0]].profile,
        )
    admitted_set = set(matrix["admitted"]) if matrix else set()

    # Verify exact 4-way parity BEFORE timing begins (unless --skip-parity).
    expected_rows: dict[str, int] = {}
    oracle_arm = (
        "duckdb_default" if "duckdb_default" in runners else next(iter(runners.keys()))
    )
    if not args.skip_parity:
        with DuckDbOracle(data_path, args.threads) as oracle_db:
            for qname in query_names:
                spec = QUERIES[qname]
                oracle_tbl = runners[oracle_arm].query(spec)
                expected_rows[qname] = oracle_tbl.num_rows
                for arm_name in arm_names:
                    if arm_name == oracle_arm:
                        continue
                    arm = ARMS[arm_name]
                    if arm.mode == "required" and qname not in admitted_set:
                        continue
                    actual_tbl = runners[arm_name].query(spec)
                    compare_tables(
                        oracle_tbl,
                        actual_tbl,
                        spec,
                        arm_name,
                        oracle_lookup=oracle_db.lookup_candidates,
                        oracle_q24=oracle_db.q24_oracle,
                    )

    # Warm page cache and verify 100% mincore residency before timing.
    initial_cache_pages = None
    if args.cache == "warm":
        warm_files(files)
        initial_cache_pages = require_warm_cache(files)
    elif args.cache == "cold":
        initial_cache_pages = cache_pages(files)

    ch_version = (
        runners["clickhouse_default"].version
        if "clickhouse_default" in runners
        else None
    )
    header = {
        "record_type": "header",
        "host": platform.node(),
        "arch": platform.machine(),
        "kernel": platform.release(),
        "cpu_affinity": (
            sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None
        ),
        "threads": args.threads,
        "cache_regime": args.cache,
        "initial_cache_pages": initial_cache_pages,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "build_revision": args.revision,
        "duckdb_version": duckdb.__version__,
        "pyarrow_version": pa.__version__,
        "clickhouse_version": ch_version,
        "dataset": summary,
        "arms": arm_names,
        "queries": query_names,
        "morsel_admitted": matrix["admitted"] if matrix else [],
        "morsel_rejections": matrix["rejections"] if matrix else {},
        "grouped_rejections": matrix["grouped_rejections"] if matrix else [],
        "duckdb_dialect_notes": DUCKDB_DIALECT_NOTES,
        "clickhouse_dialect_notes": CLICKHOUSE_DIALECT_NOTES,
    }
    print(json.dumps(header), flush=True)

    for qname in query_names:
        spec = QUERIES[qname]
        active_arms = [
            a for a in arm_names if ARMS[a].mode != "required" or qname in admitted_set
        ]
        for rep in range(1 - args.warmups, args.repeats + 1):
            order = active_arms if rep % 2 == 1 else list(reversed(active_arms))
            for arm_name in order:
                arm = ARMS[arm_name]
                before = None
                if args.cache == "cold":
                    evict_files(files)
                    before = require_cold_cache(files)
                elif rep > 0:
                    if args.cache == "warm":
                        before = require_warm_cache(files)
                    else:
                        before = cache_pages(files)
                result = runners[arm_name].measure(spec)
                validate_plan(result, arm, spec)
                if (
                    arm.engine == "datafusion"
                    and args.revision
                    and result["build_revision"] != args.revision
                ):
                    raise ValueError(
                        f"runner revision mismatch: {result['build_revision']}"
                    )
                if (
                    qname in expected_rows
                    and result["output_rows"] != expected_rows[qname]
                ):
                    raise ValueError(
                        f"{arm_name}/{qname}: unexpected output_rows "
                        f"{result['output_rows']} (expected {expected_rows[qname]})"
                    )
                if rep > 0:
                    after = (
                        require_warm_cache(files)
                        if args.cache == "warm"
                        else cache_pages(files)
                    )
                    result.pop("plan", None)
                    result.pop("build_revision", None)
                    print(
                        json.dumps(
                            {
                                "record_type": "trial",
                                "case": qname,
                                "arm": arm_name,
                                "engine": arm.engine,
                                "mode": arm.mode,
                                "profile": arm.profile,
                                "rep": rep,
                                **result,
                                "cache_pages_before": before,
                                "cache_pages_after": after,
                            }
                        ),
                        flush=True,
                    )


def summarize_jsonl(path: Path):
    lines = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    header = lines[0]
    trials = [r for r in lines[1:] if r.get("record_type") == "trial"]

    grouped: dict[tuple[str, str], list[dict]] = {}
    for t in trials:
        grouped.setdefault((t["case"], t["arm"]), []).append(t)

    medians = {
        key: statistics.median(r["seconds"] * 1000.0 for r in rows)
        for key, rows in grouped.items()
    }
    exec_medians = {
        key: statistics.median(
            r["exec_seconds"] * 1000.0
            for r in rows
            if r.get("exec_seconds") is not None
        )
        for key, rows in grouped.items()
        if any(r.get("exec_seconds") is not None for r in rows)
    }
    rows_by_query = {t["case"]: t["output_rows"] for t in trials}

    ds = header["dataset"]
    print(
        f"Host: {header['host']} ({header['arch']}, kernel {header['kernel']}), "
        f"threads={header['threads']}, cache={header['cache_regime']}, "
        f"warmups={header['warmups']}, repeats={header['repeats']}"
    )
    print(
        f"DataFusion rev: {header['build_revision']}, DuckDB: {header['duckdb_version']}, "
        f"ClickHouse: {header.get('clickhouse_version')}, PyArrow: {header['pyarrow_version']}"
    )
    print(
        f"Dataset: {ds['files']} files, {ds['rows']:,} rows, "
        f"{ds['row_groups']:,} row groups, {ds['parquet_bytes']:,} bytes"
    )
    if header.get("initial_cache_pages"):
        cp = header["initial_cache_pages"]
        print(
            f"Page cache (mincore): {cp['resident']:,}/{cp['total']:,} pages resident"
        )
    print()

    query_order = header.get("queries") or list(ALL_CLICKBENCH_QUERIES)
    arm_order = header.get("arms") or list(DEFAULT_BENCH_ARMS)

    def _fmt(q: str, a: str) -> str:
        val = medians.get((q, a))
        return f"{val:.1f}" if val is not None else "—"

    print(
        "| Query | Rows | `morsel_pushdown` ms | `stock_default` ms | `stock_pushdown` ms | "
        "`duckdb_default` ms | `clickhouse_default` ms | Morsel / Stock-PD | Morsel / DuckDB | Morsel / ClickHouse |"
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for q in query_order:
        nrows = rows_by_query.get(q)
        nrows_str = f"{nrows:,}" if nrows is not None else "—"
        m_ms = medians.get((q, "morsel_pushdown"))
        sp_ms = medians.get((q, "stock_pushdown"))
        d_ms = medians.get((q, "duckdb_default"))
        ch_ms = medians.get((q, "clickhouse_default"))
        r_sp = f"{m_ms / sp_ms:.2f}x" if (m_ms and sp_ms) else "—"
        r_d = f"{m_ms / d_ms:.2f}x" if (m_ms and d_ms) else "—"
        r_ch = f"{m_ms / ch_ms:.2f}x" if (m_ms and ch_ms) else "—"
        print(
            f"| `{q}` | {nrows_str} | {_fmt(q, 'morsel_pushdown')} | "
            f"{_fmt(q, 'stock_default')} | {_fmt(q, 'stock_pushdown')} | "
            f"{_fmt(q, 'duckdb_default')} | {_fmt(q, 'clickhouse_default')} | "
            f"{r_sp} | {r_d} | {r_ch} |"
        )

    # Geometric mean & sum across Morsel-admitted queries (all engines present).
    admitted = [q for q in query_order if all((q, a) in medians for a in arm_order)]
    if admitted and "morsel_pushdown" in arm_order:
        adm_gm = {
            a: statistics.geometric_mean(medians[(q, a)] for q in admitted)
            for a in arm_order
        }
        adm_sum = {a: sum(medians[(q, a)] for q in admitted) for a in arm_order}
        m_gm = adm_gm["morsel_pushdown"]
        m_sum = adm_sum["morsel_pushdown"]

        def _fmt_bold_adm(m: dict[str, float], key: str) -> str:
            return f"**{m[key]:.1f}**" if key in m else "—"

        def _fmt_ratio(num: float, m: dict[str, float], key: str) -> str:
            return f"**{num / m[key]:.2f}x**" if key in m else "—"

        print(
            f"| **GeoMean ({len(admitted)}/{len(query_order)} Morsel-admitted)** | — | "
            f"**{m_gm:.1f}** | "
            f"{_fmt_bold_adm(adm_gm, 'stock_default')} | "
            f"{_fmt_bold_adm(adm_gm, 'stock_pushdown')} | "
            f"{_fmt_bold_adm(adm_gm, 'duckdb_default')} | "
            f"{_fmt_bold_adm(adm_gm, 'clickhouse_default')} | "
            f"{_fmt_ratio(m_gm, adm_gm, 'stock_pushdown')} | "
            f"{_fmt_ratio(m_gm, adm_gm, 'duckdb_default')} | "
            f"{_fmt_ratio(m_gm, adm_gm, 'clickhouse_default')} |"
        )
        print(
            f"| **Sum ({len(admitted)}/{len(query_order)} Morsel-admitted)** | — | "
            f"**{m_sum:.1f}** | "
            f"{_fmt_bold_adm(adm_sum, 'stock_default')} | "
            f"{_fmt_bold_adm(adm_sum, 'stock_pushdown')} | "
            f"{_fmt_bold_adm(adm_sum, 'duckdb_default')} | "
            f"{_fmt_bold_adm(adm_sum, 'clickhouse_default')} | "
            f"{_fmt_ratio(m_sum, adm_sum, 'stock_pushdown')} | "
            f"{_fmt_ratio(m_sum, adm_sum, 'duckdb_default')} | "
            f"{_fmt_ratio(m_sum, adm_sum, 'clickhouse_default')} |"
        )

    # Geometric mean across all queries for Stock, DuckDB, and ClickHouse.
    non_morsel = [
        a
        for a in (
            "stock_default",
            "stock_pushdown",
            "duckdb_default",
            "clickhouse_default",
        )
        if a in arm_order
    ]
    shared_all = [q for q in query_order if all((q, a) in medians for a in non_morsel)]
    if shared_all and len(admitted) != len(shared_all):
        gm_map = {
            a: statistics.geometric_mean(medians[(q, a)] for q in shared_all)
            for a in non_morsel
        }
        sum_map = {a: sum(medians[(q, a)] for q in shared_all) for a in non_morsel}

        def _fmt_bold(m: dict[str, float], key: str) -> str:
            return f"**{m[key]:.1f}**" if key in m else "—"

        print(
            f"| **GeoMean ({len(shared_all)}/{len(query_order)} all queries)** | — | — | "
            f"{_fmt_bold(gm_map, 'stock_default')} | "
            f"{_fmt_bold(gm_map, 'stock_pushdown')} | "
            f"{_fmt_bold(gm_map, 'duckdb_default')} | "
            f"{_fmt_bold(gm_map, 'clickhouse_default')} | "
            f"— | — | — |"
        )
        print(
            f"| **Sum ({len(shared_all)}/{len(query_order)} all queries)** | — | — | "
            f"{_fmt_bold(sum_map, 'stock_default')} | "
            f"{_fmt_bold(sum_map, 'stock_pushdown')} | "
            f"{_fmt_bold(sum_map, 'duckdb_default')} | "
            f"{_fmt_bold(sum_map, 'clickhouse_default')} | "
            f"— | — | — |"
        )

    # Also report exec-only geomean when available across all shared queries.
    exec_arms = [
        a
        for a in arm_order
        if shared_all and all((q, a) in exec_medians for q in shared_all)
    ]
    if shared_all and exec_arms:
        print()
        parts = [
            f"`{a}`={statistics.geometric_mean(exec_medians[(q, a)] for q in shared_all):.1f} ms"
            for a in exec_arms
        ]
        print(
            f"Exec-only geometric mean ({len(shared_all)} queries): " + ", ".join(parts)
        )

    rejections = header.get("morsel_rejections") or {}
    print()
    if rejections:
        groups = header.get("grouped_rejections") or group_morsel_rejections(rejections)
        print("### Grouped Morsel (`morsel_mode=required`) Rejection Summary")
        print()
        print(
            "| Rejection category | Count | Queries | Representative planning error |"
        )
        print("| --- | ---: | --- | --- |")
        for g in groups:
            qs = ", ".join(f"`{q}`" for q in g["queries"])
            err = g["representative_error"].replace("\n", " ").replace("|", "\\|")
            print(f"| {g['category']} | {g['count']} | {qs} | `{err}` |")
    else:
        print(
            f"Morsel (`morsel_mode=required`) admission: "
            f"**{len(admitted)}/{len(query_order)} admitted (0 rejections)**."
        )


def positive_int(value: str) -> int:
    num = int(value)
    if num < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return num


def main():
    parser = argparse.ArgumentParser(
        description="ClickBench 43-query 4-way Parquet suite (Morsel vs Stock vs DuckDB vs ClickHouse)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ich = sub.add_parser(
        "install-clickhouse",
        help="Download, SHA-512 verify, and install pinned ClickHouse static binary",
    )
    ich.add_argument("--dest", type=Path, default=DEFAULT_CLICKHOUSE_DEST)
    ich.add_argument("--tag", default=CLICKHOUSE_RELEASE_TAG)
    ich.add_argument("--version", default=CLICKHOUSE_RELEASE_VERSION)
    ich.add_argument("--timeout", type=positive_int, default=300)

    dl = sub.add_parser(
        "download",
        help="Download official ClickBench Parquet files outside Git",
    )
    dl.add_argument("path", type=Path)
    dl.add_argument(
        "--count",
        type=positive_int,
        default=CLICKBENCH_PARTITIONED_TOTAL_FILES,
        help="Number of partitioned files hits_0..hits_{N-1}.parquet (default 100)",
    )
    dl.add_argument(
        "--single",
        action="store_true",
        help="Download single-file hits.parquet instead of partitioned files",
    )
    dl.add_argument("--jobs", type=positive_int, default=8)
    dl.add_argument("--timeout", type=positive_int, default=300)

    sm = sub.add_parser(
        "generate-smoke",
        help="Generate deterministic ClickBench-schema Parquet smoke fixture",
    )
    sm.add_argument("path", type=Path)
    sm.add_argument("--rows", type=positive_int, default=DEFAULT_SMOKE_ROWS)
    sm.add_argument("--files", type=positive_int, default=DEFAULT_SMOKE_FILES)
    sm.add_argument(
        "--row-group-size",
        type=positive_int,
        default=DEFAULT_SMOKE_ROW_GROUP_SIZE,
    )
    sm.add_argument("--seed", type=int, default=DEFAULT_SMOKE_SEED)

    info = sub.add_parser("info", help="Summarize a ClickBench Parquet dataset")
    info.add_argument("path", type=Path)

    pm = sub.add_parser(
        "plan-matrix",
        help="Probe morsel_mode=required admission and grouped rejections across queries",
    )
    pm.add_argument("path", type=Path)
    pm.add_argument("--runner", type=Path, required=True)
    pm.add_argument("--threads", type=positive_int, default=8)
    pm.add_argument("--timeout", type=positive_int, default=120)
    pm.add_argument("--profile", default="pushdown")
    pm.add_argument("--queries", nargs="+")

    ver = sub.add_parser(
        "verify",
        help="Verify 4-way Arrow parity and Morsel admission/rejection checks",
    )
    ver.add_argument("path", type=Path)
    ver.add_argument("--runner", type=Path, required=True)
    ver.add_argument("--clickhouse", type=Path, required=True)
    ver.add_argument("--threads", type=positive_int, default=4)
    ver.add_argument("--timeout", type=positive_int, default=120)
    ver.add_argument("--queries", nargs="+")
    ver.add_argument("--arms", nargs="+")

    bench = sub.add_parser("bench", help="Run controlled 4-way ClickBench benchmark")
    bench.add_argument("path", type=Path)
    bench.add_argument("--runner", type=Path, required=True)
    bench.add_argument("--clickhouse", type=Path, required=True)
    bench.add_argument("--revision", required=True)
    bench.add_argument("--threads", type=positive_int, default=8)
    bench.add_argument("--timeout", type=positive_int, default=300)
    bench.add_argument(
        "--cache",
        choices=("warm", "cold", "none"),
        default="warm" if platform.system() == "Linux" else "none",
    )
    bench.add_argument("--warmups", type=int, default=1)
    bench.add_argument("--repeats", type=positive_int, default=5)
    bench.add_argument("--queries", nargs="+")
    bench.add_argument("--arms", nargs="+")
    bench.add_argument("--skip-parity", action="store_true")

    summ = sub.add_parser(
        "summarize", help="Print Markdown 4-way tables from a benchmark JSONL"
    )
    summ.add_argument("jsonl", type=Path)

    dk = sub.add_parser("_duckdb-exec")
    dk.add_argument("operation", choices=("query", "measure"))
    dk.add_argument("threads", type=positive_int)
    dk.add_argument("data_path", type=Path)
    dk.add_argument("sql")

    args = parser.parse_args()
    if args.command == "install-clickhouse":
        print(
            json.dumps(
                install_clickhouse(args.dest, args.tag, args.version, args.timeout),
                indent=2,
            )
        )
        return
    if args.command == "download":
        print(
            json.dumps(
                download_dataset(
                    args.path,
                    count=args.count,
                    single=args.single,
                    jobs=args.jobs,
                    timeout=args.timeout,
                ),
                indent=2,
            )
        )
        return
    if args.command == "generate-smoke":
        print(
            json.dumps(
                generate_smoke(
                    args.path,
                    rows=args.rows,
                    files=args.files,
                    row_group_size=args.row_group_size,
                    seed=args.seed,
                ),
                indent=2,
            )
        )
        return
    if args.command == "info":
        print(json.dumps(dataset_summary(args.path), indent=2))
        return
    if args.command == "plan-matrix":
        try:
            qnames = resolve_queries(args.queries)
        except ValueError as err:
            parser.error(str(err))
        report = probe_morsel_matrix(
            args.path, args.runner, args.threads, args.timeout, qnames, args.profile
        )
        # Omit verbose plan strings from default JSON summary output.
        report.pop("plans", None)
        print(json.dumps(report, indent=2))
        return
    if args.command == "verify":
        try:
            qnames = resolve_queries(args.queries)
            arms = resolve_arms(args.arms, DEFAULT_BENCH_ARMS)
        except ValueError as err:
            parser.error(str(err))
        report = verify_suite(
            args.path,
            args.runner,
            args.clickhouse,
            args.threads,
            args.timeout,
            qnames,
            arms,
        )
        print(json.dumps(report, indent=2))
        return
    if args.command == "bench":
        if args.warmups < 0:
            parser.error("--warmups must be non-negative")
        try:
            resolve_queries(args.queries)
            resolve_arms(args.arms, DEFAULT_BENCH_ARMS)
        except ValueError as err:
            parser.error(str(err))
        run_bench(args)
        return
    if args.command == "summarize":
        summarize_jsonl(args.jsonl)
        return
    if args.command == "_duckdb-exec":
        _duckdb_exec(args.operation, args.threads, args.data_path, args.sql)
        return


if __name__ == "__main__":
    main()
