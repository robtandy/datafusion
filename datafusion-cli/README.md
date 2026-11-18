<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements
-->

<!-- Note this file is included in the crates.io page as well https://crates.io/crates/datafusion-cli -->

# Apache DataFusion Command-line Interface

[Apache DataFusion] is an extensible query execution framework, written in Rust, that uses [Apache Arrow] as its in-memory format.

DataFusion CLI (`datafusion-cli`) is a small command line utility that runs SQL queries using the DataFusion engine.

[apache arrow]: https://arrow.apache.org/
[apache datafusion]: https://datafusion.apache.org/

## Experimental morsel execution (fork only)

The `datafusion.execution.morsel_mode` session setting defaults to `off` (stock
execution). `prefer` tries the morsel engine and uses the stock planner for the
whole query if a logical operator, expression, or source is unsupported.
`required` rejects those queries with a planning error instead. Neither mode
retries native planning or execution errors using the stock engine.

```sh
DATAFUSION_EXECUTION_MORSEL_MODE=required datafusion-cli -c \
  'SHOW datafusion.execution.morsel_mode; SELECT 42;'
# SHOW reports required; SELECT 42 fails because it has no supported scan.

DATAFUSION_EXECUTION_MORSEL_MODE=prefer datafusion-cli -c 'SELECT 42;'
# The same query succeeds on the stock engine.

DATAFUSION_EXECUTION_MORSEL_MODE=off datafusion-cli -c "
  CREATE TABLE items AS VALUES (1), (2), (3);
  SET datafusion.execution.morsel_mode = 'required';
  SHOW datafusion.execution.morsel_mode;
  EXPLAIN FORMAT INDENT SELECT COUNT(*) FROM items WHERE column1 > 1;
  SELECT COUNT(*) FROM items WHERE column1 > 1;
  RESET datafusion.execution.morsel_mode;
"
# EXPLAIN shows MorselExec and CountSink; the count is 2.
```

`SET` overrides the startup value for subsequent queries in that session.
`RESET` restores the built-in default (`off`), not the startup value. An invalid
startup value fails CLI initialization. `SHOW`, `SET`, and `RESET` work in all
modes. Data-writing statements are subject to
strict planning if they execute a query (for example, `CREATE TABLE ... AS
SELECT`); create any test data requiring unsupported operators before enabling
`required`.

The selected engine applies to the optimized query, including any inlined view.
`EXPLAIN` shows the selected pipeline in physical formats; logical-only formats
still show the logical plan. `EXPLAIN ANALYZE` executes the selected pipeline.
In `required` mode both reject an unsupported query, even in a logical-only
format.

For Rust embedders, use `SessionConfig::from_env()` to load the environment once,
or set `config.options_mut().execution.morsel_mode` to a
`datafusion::common::config::MorselMode` directly.
`SessionConfig::new()` does not read the environment. Existing DataFrames retain
the session snapshot they were created with. The built-in selector uses
`datafusion.execution.target_partitions` as its native worker limit. An
explicitly installed `MorselQueryPlanner` keeps its own worker limit and remains
opted in when the mode is `off`. Other custom planners remain in charge in `off`
and `prefer`, but `SessionState` rejects data queries in `required` with such a
planner. Calling physical planners directly bypasses the session's planner
selection and custom-planner check.

# Frequently Asked Questions

## Where can I find more information?

See the [`datafusion-cli` documentation](https://datafusion.apache.org/user-guide/cli/index.html) for further information.
