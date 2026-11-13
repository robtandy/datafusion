#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
# SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

"""Tests of `measure.py`: it runs the executable of `dump_plans` that cargo
reports, wherever the target directory is.

    python3 -m unittest discover -s benchmarks/plan-compare -p 'test_*.py'
"""

from __future__ import annotations

import json
import os
import pathlib
import stat
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import measure  # noqa: E402


def artifact(name: str, executable: str | None, kind: str = "bin") -> str:
    """A `compiler-artifact` message of `cargo build --message-format=json`."""
    return json.dumps(
        {
            "reason": "compiler-artifact",
            "target": {"name": name, "kind": [kind]},
            "executable": executable,
            "fresh": True,
        }
    )


class ExecutableTest(unittest.TestCase):
    def test_the_executable_of_the_messages_of_cargo(self):
        target = "/elsewhere/target"
        messages = [
            "not a message",
            artifact("datafusion", None, "lib"),
            json.dumps({"reason": "build-script-executed"}),
            artifact(measure.DUMP_PLANS, f"{target}/release/{measure.DUMP_PLANS}"),
            json.dumps({"reason": "build-finished", "success": True}),
        ]
        self.assertEqual(
            measure.executable(messages),
            pathlib.Path(target) / "release" / measure.DUMP_PLANS,
        )

    def test_no_executable(self):
        with self.assertRaises(SystemExit):
            measure.executable([artifact("datafusion", None, "lib")])


class BuildTest(unittest.TestCase):
    """`build` with a `cargo` that builds into the target directory of
    `CARGO_TARGET_DIR`, as cargo does (a script that prints the messages of
    cargo, so that the test builds nothing)."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = pathlib.Path(self.directory.name)
        cargo = self.path / "bin" / "cargo"
        cargo.parent.mkdir()
        cargo.write_text(
            textwrap.dedent(f"""\
                #!{sys.executable}
                import json, os, sys
                arguments = sys.argv[1:]
                assert arguments[0] == "build", arguments
                assert "--message-format=json-render-diagnostics" in arguments
                manifest = arguments[arguments.index("--manifest-path") + 1]
                target = os.environ.get(
                    "CARGO_TARGET_DIR",
                    os.path.join(os.path.dirname(manifest), "target"),
                )
                executable = os.path.join(target, "release", {measure.DUMP_PLANS!r})
                print(json.dumps({{"reason": "compiler-artifact",
                                  "target": {{"name": {measure.DUMP_PLANS!r}}},
                                  "executable": executable}}))
                print(json.dumps({{"reason": "build-finished", "success": True}}))
                """),
            encoding="utf-8",
        )
        cargo.chmod(cargo.stat().st_mode | stat.S_IXUSR)
        self.environment = {"PATH": f"{cargo.parent}{os.pathsep}{os.environ['PATH']}"}

    def tearDown(self):
        self.directory.cleanup()

    def test_a_configured_target_directory(self):
        target = self.path / "configured-target"
        environment = {**self.environment, "CARGO_TARGET_DIR": str(target)}
        with mock.patch.dict(os.environ, environment):
            executable = measure.build(HERE / "dump_plans" / "Cargo.toml")
        self.assertEqual(executable, target / "release" / measure.DUMP_PLANS)

    def test_the_default_target_directory(self):
        with mock.patch.dict(os.environ, self.environment):
            os.environ.pop("CARGO_TARGET_DIR", None)
            executable = measure.build(HERE / "dump_plans" / "Cargo.toml")
        self.assertEqual(
            executable, HERE / "dump_plans" / "target" / "release" / measure.DUMP_PLANS
        )


if __name__ == "__main__":
    unittest.main()
