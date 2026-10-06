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
"""Tests the memory footprint in the per-dispatch stats of the runtime."""

import argparse
import json
import os
import struct
import subprocess
import sys
import unittest

_SCHEMA = "coralnpu.stats.v1"

# Set from the command line in main.
_RUN_MODULE = None
_MODULE = None

# Bazel test-protocol variables that must not reach the IREE tool.
_BAZEL_TEST_ENV = (
    "TEST_PREMATURE_EXIT_FILE",
    "XML_OUTPUT_FILE",
    "GTEST_OUTPUT",
    "TEST_SHARD_STATUS_FILE",
    "TEST_SHARD_INDEX",
    "TEST_TOTAL_SHARDS",
    "TEST_WARNINGS_OUTPUT_FILE",
)

_SHF_ALLOC = 0x2
_SHT_SYMTAB = 2


def _align_up(value, align):
  return (value + align - 1) & ~(align - 1) if align > 1 else value


def _c_string(data, start):
  return data[start:data.index(b"\0", start)].decode()


def _elf_layout(data, offset):
  """Returns the expected footprint of the ELF32 image at |offset|.

  Computes the ITCM and static DTCM usage from the SHF_ALLOC section headers
  (rather than the PT_LOAD segments that firmware_memory.c walks), or returns
  None if the image is not an ELF32 executable with the layout symbols.
  """
  if data[offset + 4] != 1 or data[offset + 5] != 1:  # ELFCLASS32, LSB
    return None
  (e_shoff,) = struct.unpack_from("<I", data, offset + 0x20)
  e_shentsize, e_shnum, e_shstrndx = struct.unpack_from("<HHH", data,
                                                        offset + 0x2E)
  shstr_base = offset + e_shoff + e_shstrndx * e_shentsize
  shstr_offset = offset + struct.unpack_from("<I", data, shstr_base + 16)[0]

  symbols = {}
  sections = []
  for i in range(e_shnum):
    base = offset + e_shoff + i * e_shentsize
    (
        sh_name,
        sh_type,
        sh_flags,
        sh_addr,
        sh_offset,
        sh_size,
        sh_link,
        _,
        sh_addralign,
    ) = struct.unpack_from("<IIIIIIIII", data, base)
    if sh_flags & _SHF_ALLOC and sh_size > 0:
      sections.append(
          (_c_string(data,
                     shstr_offset + sh_name), sh_addr, sh_size, sh_addralign))
    if sh_type != _SHT_SYMTAB:
      continue
    strtab_base = offset + e_shoff + sh_link * e_shentsize
    strtab_offset = offset + struct.unpack_from("<I", data, strtab_base + 16)[0]
    for sym in range(offset + sh_offset, offset + sh_offset + sh_size, 16):
      st_name, st_value = struct.unpack_from("<II", data, sym)
      symbols[_c_string(data, strtab_offset + st_name)] = st_value

  required = (
      "__itcm_start",
      "__itcm_size",
      "__dtcm_start",
      "__dtcm_size",
      "__heap_start",
      "__heap_end",
      "__stack_start",
      "__stack_size",
  )
  if any(name not in symbols for name in required):
    return None
  itcm = (symbols["__itcm_start"], symbols["__itcm_size"])
  dtcm = (symbols["__dtcm_start"], symbols["__dtcm_size"])

  def in_region(addr, size, region):
    return region[0] <= addr and addr + size <= region[0] + region[1]

  itcm_spans = [(addr, addr + size)
                for _, addr, size, _ in sections
                if in_region(addr, size, itcm)]
  dtcm_spans = [
      (addr, addr + size)
      for name, addr, size, _ in sections
      if in_region(addr, size, dtcm) and name not in (".heap", ".stack")
  ]
  heap_align = next(
      (align for name, _, _, align in sections if name == ".heap"), 1)
  itcm_used = (max(end for _, end in itcm_spans) -
               min(start for start, _ in itcm_spans) if itcm_spans else 0)
  dtcm_used = (_align_up(max(end for _, end in dtcm_spans), heap_align) -
               min(start for start, _ in dtcm_spans) if dtcm_spans else 0)

  return {
      "itcm": {
          "start": itcm[0],
          "size": itcm[1],
          "used": itcm_used,
      },
      "dtcm": {
          "start": dtcm[0],
          "size": dtcm[1],
          "used": dtcm_used,
      },
      "heap": {
          "start": symbols["__heap_start"],
          "size": symbols["__heap_end"] - symbols["__heap_start"],
      },
      "stack": {
          "start": symbols["__stack_start"],
          "size": symbols["__stack_size"],
      },
  }


def expected_layouts():
  """Returns the expected footprints of the ELFs embedded in the module."""
  with open(_MODULE, "rb") as f:
    data = f.read()
  layouts = []
  offset = data.find(b"\x7fELF")
  while offset != -1:
    layout = _elf_layout(data, offset)
    if layout is not None:
      layouts.append(layout)
    offset = data.find(b"\x7fELF", offset + 1)
  return layouts


def run_function(function, *inputs):
  """Runs |function| on MPACT and returns the stats of its dispatches."""
  env = {k: v for k, v in os.environ.items() if k not in _BAZEL_TEST_ENV}
  result = subprocess.run(
      [
          _RUN_MODULE,
          f"--module={_MODULE}",
          f"--function={function}",
          "--device=coralnpu",
          "--simulator=mpact",
          "--coralnpu_stats_format=json",
          "--coralnpu_stats_file=-",
          *[f"--input={value}" for value in inputs],
      ],
      capture_output=True,
      text=True,
      check=False,
      env=env,
  )
  if result.returncode != 0:
    raise AssertionError(
        f"{function} failed ({result.returncode}):\n{result.stderr}")
  prefix = '{"schema":"%s"' % _SCHEMA
  records = [
      json.loads(line)
      for line in result.stdout.splitlines()
      if line.startswith(prefix)
  ]
  if not records:
    raise AssertionError(f"{function} reported no stats:\n{result.stdout}")
  return records


class MemoryTest(unittest.TestCase):

  def assert_program_layout(self, memory):
    """Checks the ITCM/DTCM/heap/stack layout against the embedded ELFs."""
    actual = {
        "itcm": memory["itcm"],
        "dtcm": memory["dtcm"],
        "heap": {
            "start": memory["heap"]["start"],
            "size": memory["heap"]["size"],
        },
        "stack": memory["stack"],
    }
    self.assertIn(actual, expected_layouts())
    # The configured regions of the crt linker script.
    self.assertEqual(memory["itcm"]["start"], 0)
    self.assertEqual(memory["itcm"]["size"], 8 * 1024)
    self.assertEqual(memory["dtcm"]["start"], 0x10000)
    self.assertEqual(memory["dtcm"]["size"], 32 * 1024)
    self.assertEqual(memory["stack"]["size"], 4096)
    self.assertGreater(memory["itcm"]["used"], 0)
    self.assertLessEqual(memory["itcm"]["used"], memory["itcm"]["size"])
    self.assertGreater(memory["dtcm"]["used"], 0)
    self.assertLessEqual(memory["heap"]["used"], memory["heap"]["size"])

  def test_static(self):
    records = run_function("add", "1000xi32=1", "1000xi32=2")
    self.assertEqual(len(records), 1, records)
    record = records[0]
    self.assertEqual(record["status"], {"code": "OK"}, record)
    memory = record["memory"]
    self.assert_program_layout(memory)
    # No push constants, and the binding address and length tables of the two
    # inputs and the result, packed without padding.
    self.assertEqual(memory["heap"]["used"], 2 * 3 * 4)

  def test_dynamic(self):
    records = run_function("add_dynamic", "100xi32=1", "100xi32=2")
    for record in records:
      self.assertEqual(record["status"], {"code": "OK"}, record)
      memory = record["memory"]
      self.assert_program_layout(memory)
      # The dynamic size is passed as push constants, which come on top of the
      # binding tables of the three bindings.
      self.assertGreater(memory["heap"]["used"], 2 * 3 * 4)
      self.assertEqual(memory["heap"]["used"] % 4, 0)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--run-module", required=True)
  parser.add_argument("--module", required=True)
  args, unittest_args = parser.parse_known_args()
  _RUN_MODULE = args.run_module
  _MODULE = args.module
  unittest.main(argv=[sys.argv[0]] + unittest_args)
