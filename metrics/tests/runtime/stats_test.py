# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests the per-dispatch execution stats of the CoralNPU runtime."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import unittest

_SCHEMA = "coralnpu.stats.v1"

# Set from the command line in main.
_CHECK_MODULE = None
_MODULE = None

# Bazel test-protocol variables that must not reach iree-check-module: it is
# a gtest binary, so it would create TEST_PREMATURE_EXIT_FILE (left behind
# when it exits early, e.g. on a bad flag) and overwrite this test's XML
# output.
_BAZEL_TEST_ENV = (
    "TEST_PREMATURE_EXIT_FILE",
    "XML_OUTPUT_FILE",
    "GTEST_OUTPUT",
    "TEST_SHARD_STATUS_FILE",
    "TEST_SHARD_INDEX",
    "TEST_TOTAL_SHARDS",
    "TEST_WARNINGS_OUTPUT_FILE",
)


def run_check_module(*flags, **extra_env):
  """Runs the test module with iree-check-module on MPACT."""
  env = {k: v for k, v in os.environ.items() if k not in _BAZEL_TEST_ENV}
  env.update(extra_env)
  return subprocess.run(
      [
          _CHECK_MODULE,
          f"--module={_MODULE}",
          "--device=coralnpu",
          "--simulator=mpact",
          *flags,
      ],
      capture_output=True,
      text=True,
      check=False,
      env=env,
  )


def json_records(text):
  """Returns the stats records in |text|, skipping unrelated lines."""
  prefix = '{"schema":"%s"' % _SCHEMA
  return [
      json.loads(line) for line in text.splitlines() if line.startswith(prefix)
  ]


class StatsTest(unittest.TestCase):

  def assert_valid_record(self, record):
    self.assertEqual(record["schema"], _SCHEMA)
    self.assertTrue(record["dispatch"]["executable"])
    self.assertTrue(record["dispatch"]["name"])
    self.assertIsInstance(record["dispatch"]["ordinal"], int)
    self.assertEqual(record["backend"], {"simulator": "mpact"})
    self.assertEqual(record["status"], {"code": "OK"})
    self.assertGreater(record["cycles"]["total"], 0)

  def test_json_file(self):
    with tempfile.TemporaryDirectory(dir=os.environ.get("TEST_TMPDIR")) as tmp:
      path = os.path.join(tmp, "stats.jsonl")
      result = run_check_module("--coralnpu_stats_format=json",
                                f"--coralnpu_stats_file={path}")
      self.assertEqual(result.returncode, 0, result.stderr)
      with open(path) as f:
        lines = f.read().splitlines()
    records = [json.loads(line) for line in lines]
    # Every line is a record, and both functions' dispatches are present even
    # though each function runs on its own device.
    self.assertEqual(len(records), len(lines))
    for record in records:
      self.assert_valid_record(record)
    names = {record["dispatch"]["name"] for record in records}
    self.assertGreaterEqual(len(names), 2, records)

  def test_json_stdout(self):
    result = run_check_module("--coralnpu_stats_format=json",
                              "--coralnpu_stats_file=-")
    self.assertEqual(result.returncode, 0, result.stderr)
    records = json_records(result.stdout)
    self.assertGreaterEqual(len(records), 2, result.stdout)
    for record in records:
      self.assert_valid_record(record)

  def test_pretty(self):
    result = run_check_module("--coralnpu_stats_format=pretty")
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertIn("CoralNPU dispatch stats (backend: mpact)", result.stderr)
    self.assertIn("executable::export", result.stderr)
    self.assertIn("(total)", result.stderr)

  def test_disabled_by_default(self):
    result = run_check_module()
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertNotIn(_SCHEMA, result.stdout + result.stderr)
    self.assertNotIn("CoralNPU dispatch stats", result.stdout + result.stderr)

  def test_env_fallback(self):
    # The environment applies unless the flags are set.
    result = run_check_module(CORALNPU_STATS_FORMAT="json",
                              CORALNPU_STATS_FILE="-")
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertGreaterEqual(len(json_records(result.stdout)), 2, result.stdout)
    result = run_check_module("--coralnpu_stats_format=pretty",
                              CORALNPU_STATS_FORMAT="json")
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertIn("CoralNPU dispatch stats", result.stderr)

  def test_unknown_format(self):
    result = run_check_module("--coralnpu_stats_format=xml")
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("unknown CoralNPU stats format 'xml'",
                  result.stdout + result.stderr)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--check-module", required=True)
  parser.add_argument("--module", required=True)
  args, unittest_args = parser.parse_known_args()
  _CHECK_MODULE = args.check_module
  _MODULE = args.module
  unittest.main(argv=[sys.argv[0]] + unittest_args)
