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
import struct
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


def run_check_module(*flags, module=None, **extra_env):
  """Runs the test module with iree-check-module on MPACT."""
  env = {k: v for k, v in os.environ.items() if k not in _BAZEL_TEST_ENV}
  env.update(extra_env)
  return subprocess.run(
      [
          _CHECK_MODULE,
          f"--module={module or _MODULE}",
          "--device=coralnpu",
          "--simulator=mpact",
          *flags,
      ],
      capture_output=True,
      text=True,
      check=False,
      env=env,
  )


def _halt_at_entry(module_bytes):
  """Replaces the first instruction of each embedded ELF with mpause."""
  data = bytearray(module_bytes)
  offset = data.find(b"\x7fELF")
  while offset != -1:
    if data[offset + 4:offset + 6] == b"\x01\x01":
      (e_phoff,) = struct.unpack_from("<I", data, offset + 0x1C)
      e_phentsize, e_phnum = struct.unpack_from("<HH", data, offset + 0x2A)
      for i in range(e_phnum):
        phdr = offset + e_phoff + i * e_phentsize
        p_type, p_offset, p_vaddr, _, p_filesz = struct.unpack_from(
            "<IIIII", data, phdr)
        if p_type == 1 and p_vaddr == 0 and p_filesz >= 4:
          struct.pack_into("<I", data, offset + p_offset, 0x08000073)  # mpause
    offset = data.find(b"\x7fELF", offset + 1)
  return bytes(data)


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
    # Firmware-measured counters of the workgroup loop only.
    self.assertGreater(record["cycles"]["kernel"], 0)
    self.assertLess(record["cycles"]["kernel"], record["cycles"]["total"])
    self.assertGreater(record["instret"]["kernel"], 0)

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
    self.assertIn("kcycles(total)", result.stderr)
    self.assertIn("kinstret(total)", result.stderr)
    totals = [
        line.split()
        for line in result.stderr.splitlines()
        if line.endswith("  (total)")
    ]
    self.assertTrue(totals, result.stderr)
    for row in totals:
      _, _, cycles, kcycles, kinstret = (int(x) for x in row[:5])
      self.assertGreater(kcycles, 0)
      self.assertLess(kcycles, cycles)
      self.assertGreater(kinstret, 0)

  def test_disabled_by_default(self):
    result = run_check_module()
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertNotIn(_SCHEMA, result.stdout + result.stderr)
    self.assertNotIn("CoralNPU dispatch stats", result.stdout + result.stderr)

  def test_failed_dispatch_omits_kernel_counters(self):
    with tempfile.TemporaryDirectory(dir=os.environ.get("TEST_TMPDIR")) as tmp:
      bad_module = os.path.join(tmp, "bad_module.vmfb")
      with open(_MODULE, "rb") as f:
        patched = _halt_at_entry(f.read())
      with open(bad_module, "wb") as f:
        f.write(patched)
      result = run_check_module(
          "--coralnpu_stats_format=json",
          "--coralnpu_stats_file=-",
          module=bad_module,
      )
      self.assertNotEqual(result.returncode, 0)
      records = json_records(result.stdout)
      self.assertTrue(records, result.stdout)
      for record in records:
        self.assertNotEqual(record["status"]["code"], "OK")
        self.assertGreater(record["cycles"]["total"], 0)
        self.assertNotIn("kernel", record["cycles"])
        self.assertNotIn("instret", record)
      pretty = run_check_module("--coralnpu_stats_format=pretty",
                                module=bad_module)
      self.assertNotEqual(pretty.returncode, 0)
      totals = [
          line.split()
          for line in pretty.stderr.splitlines()
          if line.endswith("  (total)")
      ]
      self.assertTrue(totals, pretty.stderr)
      for row in totals:
        self.assertGreater(int(row[1]), 0)  # failed
        self.assertGreater(int(row[2]), 0)  # cycles(total)
        self.assertEqual(row[3], "-")  # kcycles(total)
        self.assertEqual(row[4], "-")  # kinstret(total)

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
