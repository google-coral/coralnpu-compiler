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
"""Tests the workload report (metrics/tools/report.py) on synthetic inputs."""

import json
import os
import tempfile
import unittest

from metrics.tools import report
from metrics.tools import workload_report

_TARGET = {"triple": "riscv32", "vlen": 128, "zvt": True, "dtcm_kb": 32}


def make_bundle(executable, dispatches, itcm_used=1000, dtcm_used=100):
  """Returns a compiler metrics bundle of |executable|.

  |dispatches| are (name, candidate, lowered) tuples.
  """
  return {
      "schema": "coralnpu.metrics.v1",
      "executable": executable,
      "target": _TARGET,
      "dispatches": [{
          "name": name,
          "tile_grid": {
              "op": "linalg.matmul",
              "loops": [64, 96, 128],
              "workgroup": [64, 96, 128],
              "dtcm": [64, 96, 128],
              "vector": [32, 32, 1],
              **({
                  "zvt_grid": [2, 2]
              } if lowered else {}),
          },
          "zvt": {
              "candidate": candidate,
              "lowered": lowered
          },
      } for name, candidate, lowered in dispatches],
      "elf": {
          "regions": {
              "itcm": {
                  "size": 8192,
                  "used": itcm_used
              },
              "dtcm": {
                  "size": 32768,
                  "used": dtcm_used
              },
          }
      },
  }


def make_record(executable, name, code="OK", simulator="mpact", **values):
  """Returns a runtime stats record; |values| are its optional keys."""
  record = {
      "schema": "coralnpu.stats.v1",
      "dispatch": {
          "executable": executable,
          "name": name,
          "ordinal": 0
      },
      "backend": {
          "simulator": simulator
      },
      "status": {
          "code": code
      },
  }
  record.update(values)
  return record


class ReportTest(unittest.TestCase):

  def test_aggregates_runs(self):
    """Tests per-dispatch and simulator-level aggregation across multiple runs."""
    bundles = [make_bundle("exe", [("d", True, True)])]
    records = [
        make_record("exe",
                    "d",
                    cycles={
                        "total": 100,
                        "kernel": 80
                    },
                    instret={"kernel": 70},
                    memory={
                        "heap": {
                            "used": 64
                        },
                        "stack": {
                            "peak": 200,
                            "overflow": False
                        }
                    }),
        # A second successful run without instret.kernel omits instret_per_run.
        make_record("exe", "d", cycles={
            "total": 120,
            "kernel": 100
        }),
        # A failed run reports no kernel counters.
        make_record("exe",
                    "d",
                    code="INTERNAL",
                    cycles={"total": 50},
                    memory={
                        "heap": {
                            "used": 32
                        },
                        "stack": {
                            "peak": 300,
                            "overflow": True
                        }
                    }),
    ]
    merged = report.build_report(bundles, records)
    self.assertEqual(merged["target"]["itcm_kb"], 8)
    self.assertEqual(merged["summary"]["firmware"], {
        "executables": 1,
        "itcm_used": 1000,
        "dtcm_used": 100,
    })
    self.assertEqual(merged["dispatches"][0]["itcm_used"], 1000)
    self.assertEqual(merged["dispatches"][0]["data_used"], 100)
    run = merged["dispatches"][0]["runs"]["mpact"]
    self.assertEqual(run["runs"], 3)
    self.assertEqual(run["failures"], 1)
    # Only the total cycles of all the runs have a sum; per-run means are over
    # the successful runs when all of them reported the counter.
    self.assertEqual(run["cycles"], {"total": 270})
    self.assertEqual(run["cycles_per_run"], {"total": 110, "kernel": 90})
    self.assertNotIn("instret_per_run", run)
    self.assertEqual(run["memory"], {
        "heap_used": 64,
        "stack_peak": 300,
        "stack_overflow": True
    })
    summary = merged["summary"]["simulators"]["mpact"]
    self.assertEqual(
        summary, {
            "runs": 3,
            "failures": 1,
            "cycles": {
                "total": 270
            },
            "memory": {
                "heap_used": 64,
                "stack_peak": 300,
                "stack_overflow": True,
            },
        })

  def test_zvt_rate(self):
    """Tests static Zvt candidate and lowering rate calculation."""
    bundles = [
        make_bundle("a", [("lowered", True, True), ("missed", True, False)]),
        make_bundle("b", [("rvv", False, False), ("unexpected", False, True)]),
    ]
    records = [
        make_record("a", "lowered"),
        make_record("a", "lowered"),
        make_record("a", "missed"),
        make_record("b", "rvv"),
    ]
    merged = report.build_report(bundles, records)
    self.assertEqual(merged["summary"]["zvt"], {
        "candidates": 2,
        "lowered_candidates": 1,
        "rate": 0.5
    })

  def test_problems(self):
    """Tests detection of missing bundles, missing simulator stats, and failures."""
    bundle = make_bundle("exe", [("d", True, True)])
    # A bundled non-Linalg or dynamic dispatch has no tile_grid.
    bundle["dispatches"].append({"name": "copy"})
    records = [
        make_record("exe", "d", code="INTERNAL"),
        make_record("exe",
                    "copy",
                    memory={"stack": {
                        "peak": 4096,
                        "overflow": True
                    }}),
        make_record("other",
                    "e",
                    memory={
                        "itcm": {
                            "used": 500
                        },
                        "dtcm": {
                            "used": 50
                        }
                    }),
    ]
    merged = report.build_report([bundle], records)
    # The dispatch without a bundle comes last and falls back to the runtime
    # ITCM and DTCM values.
    self.assertEqual([dispatch["name"] for dispatch in merged["dispatches"]],
                     ["d", "copy", "e"])
    self.assertEqual(merged["dispatches"][2]["itcm_used"], 500)
    self.assertEqual(merged["dispatches"][2]["data_used"], 50)
    with self.assertRaises(ValueError):
      report.build_report(
          [bundle], [make_record("exe", "d", memory={"itcm": {
              "used": 999
          }})])
    self.assertEqual(report.find_problems(merged, ["mpact", "verilator"]), [
        "no runtime stats from verilator",
        "exe::d failed 1 of 1 runs on mpact",
        "exe::copy overflowed the stack on mpact",
        "no compiler metrics for dispatch other::e",
    ])
    self.assertEqual(report.find_problems(report.build_report([], []), []),
                     ["no compiler metrics bundle"])

  def test_pretty(self):
    """Tests Markdown report formatting across summary and tables."""
    bundles = [
        make_bundle(
            "exe",
            [("lowered", True, True), ("missed", True, False),
             ("rvv", False, False)],
            itcm_used=2000,
            dtcm_used=150,
        )
    ]
    records = [
        make_record(
            "exe",
            "lowered",
            cycles={
                "total": 100,
                "kernel": 80
            },
            instret={"kernel": 70},
            memory={
                "heap": {
                    "used": 4096
                },
                "stack": {
                    "peak": 1024,
                    "overflow": True
                },
            },
        ),
    ]
    merged = report.build_report(bundles, records)
    self.assertEqual(merged["summary"]["simulators"]["mpact"]["instret"],
                     {"kernel": 70})
    text = report.format_pretty(merged)
    self.assertIn("- Target: riscv32, VLEN 128, Zvt yes, ITCM 8 KB, DTCM 32 KB",
                  text)
    self.assertIn(
        "- Static Zvt rate: 1 of 2 Zvt candidates lowered to Zvt "
        "(50.0%)\n", text)
    self.assertIn(
        "- mpact: 1 dispatch runs, 0 failed, 100 cycles, 80 in the kernels\n",
        text)
    self.assertIn("- Firmware: up to 2000 bytes of ITCM and 150 bytes of DTCM",
                  text)
    for section in ("## Dispatches", "## Tile grid"):
      self.assertIn(section, text)
    self.assertNotIn("## Executables", text)
    lines = text.splitlines()

    def row(*prefix):
      """Returns the cells of the first table row that starts with |prefix|."""
      return next(line.split()
                  for line in lines
                  if line.split()[:len(prefix)] == list(prefix))

    self.assertEqual(row("lowered", "mpact"), [
        "lowered", "mpact", "1", "0", "100", "80", "70", "2000", "150", "4096",
        "1024!", "yes"
    ])
    # Dispatches that did not run have n/a runtime columns, static itcm/data,
    # and merged zvt status ("yes/no" when candidate != lowered, "no" when both
    # are False).
    self.assertEqual(row("missed", "n/a"), ["missed"] + ["n/a"] * 6 +
                     ["2000", "150", "n/a", "n/a", "yes/no"])
    self.assertEqual(row("rvv", "n/a"), ["rvv"] + ["n/a"] * 6 +
                     ["2000", "150", "n/a", "n/a", "no"])
    self.assertEqual(row("lowered", "linalg.matmul"), [
        "lowered", "linalg.matmul", "64x96x128", "64x96x128", "64x96x128",
        "32x32x1", "2x2"
    ])
    # A non-Zvt tile grid row omits the zvt column instead of printing n/a.
    self.assertEqual(row("missed", "linalg.matmul"), [
        "missed", "linalg.matmul", "64x96x128", "64x96x128", "64x96x128",
        "32x32x1"
    ])
    # Colliding dispatch names across executables are qualified as exe::name.
    colliding = report.format_pretty(
        report.build_report([
            make_bundle("exe0", [("d", True, True)]),
            make_bundle("exe1", [("d", False, False)]),
        ], []))
    self.assertIn("exe0::d", colliding)
    self.assertIn("exe1::d", colliding)

  def test_main(self):
    """Tests CLI loading of concatenated bundles and JSON Lines stats."""
    with tempfile.TemporaryDirectory() as tmp:
      bundles_path = os.path.join(tmp, "bundles.json")
      # The bundles one after the other, as with --coralnpu-dump-metrics-dir=-.
      with open(bundles_path, "w") as f:
        f.write(json.dumps(make_bundle("a", [("d", True, True)]), indent=2))
        f.write("\n")
        f.write(json.dumps(make_bundle("b", [("e", False, False)]), indent=2))
      stats_path = os.path.join(tmp, "stats.jsonl")
      with open(stats_path, "w") as f:
        f.write(json.dumps(make_record("a", "d")) + "\n")
      output_path = os.path.join(tmp, "report.json")
      self.assertEqual(
          report.main([
              f"--compiler-metrics={bundles_path}",
              f"--runtime-stats={stats_path}",
              "--format=json",
              f"--output={output_path}",
          ]), 0)
      with open(output_path) as f:
        merged = json.load(f)
      self.assertEqual(
          report.main([f"--compiler-metrics={os.path.join(tmp, 'missing')}"]),
          2)
    self.assertEqual(merged["schema"], "coralnpu.report.v1")
    self.assertEqual(merged["simulators"], ["mpact"])
    self.assertEqual(
        [executable["executable"] for executable in merged["executables"]],
        ["a", "b"])
    self.assertEqual(merged["dispatches"][0]["runs"]["mpact"]["runs"], 1)

  def test_workload_report(self):
    """Tests workload_report exit codes, stale stats cleanup, and outputs."""
    with tempfile.TemporaryDirectory() as tmp:
      bundles_path = os.path.join(tmp, "bundles.json")
      with open(bundles_path, "w") as f:
        json.dump(make_bundle("a", [("d", True, True)]), f)
      runner_path = os.path.join(tmp, "fake_run_module.sh")
      record = json.dumps(make_record("a", "d"))
      with open(runner_path, "w") as f:
        f.write("#!/bin/sh\n"
                "for arg in \"$@\"; do\n"
                "  case \"$arg\" in\n"
                "    --coralnpu_stats_file=*) file=\"${arg#*=}\" ;;\n"
                "    --fail) fail=1 ;;\n"
                "  esac\n"
                "done\n"
                f"printf '%s\\n' '{record}' >> \"$file\"\n"
                "exit ${fail:-0}\n")
      os.chmod(runner_path, 0o755)
      # Unreadable bundle returns 2.
      self.assertEqual(
          workload_report.main([
              f"--run-module={runner_path}",
              "--module=m.vmfb",
              f"--compiler-metrics={os.path.join(tmp, 'missing')}",
              f"--output-dir={tmp}",
          ]), 2)
      # Pre-populate a stale unbundled dispatch in stats_mpact.jsonl;
      # run_simulator must remove it before running.
      stats_path = os.path.join(tmp, "stats_mpact.jsonl")
      with open(stats_path, "w") as f:
        f.write(json.dumps(make_record("stale", "x")) + "\n")
      self.assertEqual(
          workload_report.main([
              f"--run-module={runner_path}",
              "--module=m.vmfb",
              f"--compiler-metrics={bundles_path}",
              f"--output-dir={tmp}",
          ]), 0)
      # A failed simulator run returns 1 while still writing report.md and
      # report.json.
      os.remove(os.path.join(tmp, "report.md"))
      os.remove(os.path.join(tmp, "report.json"))
      self.assertEqual(
          workload_report.main([
              f"--run-module={runner_path}",
              "--module=m.vmfb",
              f"--compiler-metrics={bundles_path}",
              "--flag=--fail",
              f"--output-dir={tmp}",
          ]), 1)
      self.assertTrue(os.path.exists(os.path.join(tmp, "report.md")))
      self.assertTrue(os.path.exists(os.path.join(tmp, "report.json")))

  def test_load_bundles_from_directory(self):
    """Tests loading *.metrics.json files in sorted order from a directory."""
    with tempfile.TemporaryDirectory() as tmp:
      for executable in ("b", "a"):
        with open(os.path.join(tmp, f"{executable}.metrics.json"), "w") as f:
          json.dump(make_bundle(executable, []), f)
      # Other files and other schemas are not bundles.
      with open(os.path.join(tmp, "notes.txt"), "w") as f:
        f.write("not a bundle")
      with open(os.path.join(tmp, "c.metrics.json"), "w") as f:
        json.dump({"schema": "other"}, f)
      bundles = report.load_bundles([tmp])
      with self.assertRaises(ValueError):
        report.load_bundles([tmp, os.path.join(tmp, "a.metrics.json")])
    self.assertEqual([bundle["executable"] for bundle in bundles], ["a", "b"])


if __name__ == "__main__":
  unittest.main()
