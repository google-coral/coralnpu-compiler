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


def _shrink_stack(module_bytes, stack_size):
  """Shrinks the declared stack of each embedded ELF to |stack_size| bytes."""
  data = bytearray(module_bytes)
  offset = data.find(b"\x7fELF")
  while offset != -1:
    if data[offset + 4:offset + 6] == b"\x01\x01":
      (e_shoff,) = struct.unpack_from("<I", data, offset + 0x20)
      e_shentsize, e_shnum = struct.unpack_from("<HH", data, offset + 0x2E)
      for i in range(e_shnum):
        shdr = offset + e_shoff + i * e_shentsize
        _, sh_type, _, _, sh_offset, sh_size, sh_link = struct.unpack_from(
            "<IIIIIII", data, shdr)
        if sh_type != 2:  # SHT_SYMTAB
          continue
        strtab = offset + e_shoff + sh_link * e_shentsize
        strtab_off = offset + struct.unpack_from("<I", data, strtab + 16)[0]
        sym_offsets = {}
        sym_values = {}
        for sym in range(offset + sh_offset, offset + sh_offset + sh_size, 16):
          st_name, st_value = struct.unpack_from("<II", data, sym)
          end = data.index(b"\0", strtab_off + st_name)
          name = data[strtab_off + st_name:end].decode()
          sym_offsets[name] = sym
          sym_values[name] = st_value
        if "__stack_start" in sym_offsets and "__stack_size" in sym_offsets:
          stack_end = sym_values["__stack_start"] + sym_values["__stack_size"]
          new_start = stack_end - stack_size
          struct.pack_into("<I", data, sym_offsets["__heap_end"] + 4, new_start)
          struct.pack_into("<I", data, sym_offsets["__stack_start"] + 4,
                           new_start)
          struct.pack_into("<I", data, sym_offsets["__stack_size"] + 4,
                           stack_size)
    offset = data.find(b"\x7fELF", offset + 1)
  return bytes(data)


def json_records(text):
  """Returns the stats records in |text|, skipping unrelated lines."""
  prefix = '{"schema":"%s"' % _SCHEMA
  return [
      json.loads(line) for line in text.splitlines() if line.startswith(prefix)
  ]


class StatsTest(unittest.TestCase):

  def assert_valid_record(self, record, stack):
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
    # The memory footprint comes from the host (memory_test checks it in
    # detail).
    memory = record["memory"]
    self.assertGreater(memory["itcm"]["used"], 0)
    self.assertLessEqual(memory["itcm"]["used"], memory["itcm"]["size"])
    self.assertGreater(memory["heap"]["used"], 0)
    self.assertLessEqual(memory["heap"]["used"], memory["heap"]["size"])
    # Opt-in, measured by the host (stack_peak_test checks the measurement).
    if stack:
      self.assertGreater(memory["stack"]["peak"], 0)
      self.assertLess(memory["stack"]["peak"], memory["stack"]["size"])
      self.assertIs(memory["stack"]["overflow"], False)
    else:
      self.assertNotIn("peak", memory["stack"])
      self.assertNotIn("overflow", memory["stack"])

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
      self.assert_valid_record(record, stack=False)
    names = {record["dispatch"]["name"] for record in records}
    self.assertGreaterEqual(len(names), 2, records)

  def test_json_stdout(self):
    result = run_check_module(
        "--coralnpu_stats_format=json",
        "--coralnpu_stats_file=-",
        "--coralnpu_stats_collect=stack",
    )
    self.assertEqual(result.returncode, 0, result.stderr)
    records = json_records(result.stdout)
    self.assertGreaterEqual(len(records), 2, result.stdout)
    for record in records:
      self.assert_valid_record(record, stack=True)

  def test_pretty(self):
    result = run_check_module("--coralnpu_stats_format=pretty")
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertIn("CoralNPU dispatch stats (backend: mpact)", result.stderr)
    self.assertIn("executable::export", result.stderr)
    self.assertIn("(total)", result.stderr)
    self.assertIn("kcycles(total)", result.stderr)
    self.assertIn("kinstret(total)", result.stderr)
    self.assertIn("itcm(max)", result.stderr)
    self.assertIn("data(max)", result.stderr)
    self.assertIn("heap(max)", result.stderr)
    self.assertIn("stack(max)", result.stderr)
    totals = [
        line.split()
        for line in result.stderr.splitlines()
        if line.endswith("  (total)")
    ]
    self.assertTrue(totals, result.stderr)
    for row in totals:
      _, _, cycles, kcycles, kinstret, itcm, data, heap = (
          int(x) for x in row[:8])
      self.assertGreater(kcycles, 0)
      self.assertLess(kcycles, cycles)
      self.assertGreater(kinstret, 0)
      self.assertGreater(itcm, 0)
      self.assertGreater(data, 0)
      self.assertGreater(heap, 0)
      self.assertEqual(row[8], "-")
    result = run_check_module("--coralnpu_stats_format=pretty",
                              "--coralnpu_stats_collect=stack")
    self.assertEqual(result.returncode, 0, result.stderr)
    for line in result.stderr.splitlines():
      if line.endswith("  (total)"):
        self.assertGreater(int(line.split()[8]), 0)

  def test_stack_overflow(self):
    with tempfile.TemporaryDirectory(dir=os.environ.get("TEST_TMPDIR")) as tmp:
      small_stack = os.path.join(tmp, "small_stack.vmfb")
      with open(_MODULE, "rb") as f:
        patched = _shrink_stack(f.read(), stack_size=4)
      with open(small_stack, "wb") as f:
        f.write(patched)
      result = run_check_module(
          "--coralnpu_stats_format=json",
          "--coralnpu_stats_file=-",
          "--coralnpu_stats_collect=stack",
          module=small_stack,
      )
      self.assertEqual(result.returncode, 0, result.stderr)
      records = json_records(result.stdout)
      self.assertTrue(records, result.stdout)
      for record in records:
        stack = record["memory"]["stack"]
        self.assertEqual(stack["size"], 4)
        self.assertGreater(stack["peak"], 4)
        self.assertIs(stack["overflow"], True)
      pretty = run_check_module(
          "--coralnpu_stats_format=pretty",
          "--coralnpu_stats_collect=stack",
          module=small_stack,
      )
      self.assertEqual(pretty.returncode, 0, pretty.stderr)
      totals = [
          line.split()
          for line in pretty.stderr.splitlines()
          if line.endswith("  (total)")
      ]
      self.assertTrue(totals, pretty.stderr)
      for row in totals:
        self.assertTrue(row[8].endswith("!"), row)
        self.assertGreater(int(row[8][:-1]), 4)

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
                              CORALNPU_STATS_FILE="-",
                              CORALNPU_STATS_COLLECT="stack")
    self.assertEqual(result.returncode, 0, result.stderr)
    records = json_records(result.stdout)
    self.assertGreaterEqual(len(records), 2, result.stdout)
    for record in records:
      self.assert_valid_record(record, stack=True)
    result = run_check_module("--coralnpu_stats_format=pretty",
                              CORALNPU_STATS_FORMAT="json")
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertIn("CoralNPU dispatch stats", result.stderr)

  def test_unknown_format(self):
    result = run_check_module("--coralnpu_stats_format=xml")
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("unknown CoralNPU stats format 'xml'",
                  result.stdout + result.stderr)

  def test_unknown_collector(self):
    for flags in (
        ("--coralnpu_stats_format=json", "--coralnpu_stats_collect=stack,heap"),
        ("--coralnpu_stats_collect=stack,heap",),
    ):
      result = run_check_module(*flags)
      self.assertNotEqual(result.returncode, 0)
      self.assertIn("unknown CoralNPU stats collector 'heap'",
                    result.stdout + result.stderr)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--check-module", required=True)
  parser.add_argument("--module", required=True)
  args, unittest_args = parser.parse_known_args()
  _CHECK_MODULE = args.check_module
  _MODULE = args.module
  unittest.main(argv=[sys.argv[0]] + unittest_args)
