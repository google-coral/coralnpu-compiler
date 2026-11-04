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
"""Workload report: runs a CoralNPU module and reports its metrics.

For one workload, a module compiled for CoralNPU together with its compiler
metrics bundles (iree-compile --coralnpu-dump-metrics-format=json), this
tool runs a function with iree-run-module on each simulator, with the runtime
stats (--coralnpu_stats_format=json and --coralnpu_stats_collect=stack), and
merges the bundles and the stats with report.py.

The iree-run-module flags (--flag, e.g. --device=coralnpu) and the inputs
(--input, iree-run-module --input values) are passed through; the tool adds
only the function, the simulator and the stats flags.

It prints the report as markdown and writes these files to --output-dir, by
default $TEST_UNDECLARED_OUTPUTS_DIR when it runs under `bazel test`:
  report.md                the report, as printed
  report.json              the report (schema "coralnpu.report.v1")
  stats_<simulator>.jsonl  the runtime stats of each simulator

Exits with status 1 if a run fails or if data is missing (no bundle, no stats
from a simulator, or a dispatch that ran without a bundle), and 2 if the
bundles can't be read. Never fails on performance.

Usage:
  workload_report.py --run-module iree-run-module --module model.vmfb \\
      --compiler-metrics model.metrics.json --flag=--device=coralnpu \\
      --simulator mpact --input 16x64xf32=1 --input 64x32xf32=0.5
"""

import argparse
import json
import os
import subprocess
import sys
import time

from metrics.tools import report

# The opt-in collectors of the runtime stats.
_STATS_COLLECT = "stack"


def run_simulator(args, simulator):
  """Runs the module on |simulator|; returns its stats records and problems."""
  stats_path = os.path.join(args.output_dir, f"stats_{simulator}.jsonl")
  # The runtime appends to the stats file: remove the stats of an earlier
  # run.
  if os.path.exists(stats_path):
    os.remove(stats_path)
  command = [
      args.run_module,
      f"--module={args.module}",
      f"--function={args.function}",
      f"--simulator={simulator}",
      "--coralnpu_stats_format=json",
      f"--coralnpu_stats_file={stats_path}",
      f"--coralnpu_stats_collect={_STATS_COLLECT}",
  ] + args.flag + [f"--input={value}" for value in args.input]
  env = {
      k: v
      for k, v in os.environ.items()
      if k != "CORALNPU_SIMULATOR" and not k.startswith("CORALNPU_STATS_")
  }
  start = time.monotonic()
  process = subprocess.run(command,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT,
                           text=True,
                           check=False,
                           env=env)
  print(
      f"{simulator}: ran {args.function} in "
      f"{time.monotonic() - start:.1f} s",
      file=sys.stderr)
  problems = []
  # The output of a failed run tells why it failed.
  if process.returncode != 0:
    problems.append(f"{' '.join(command)} failed with exit status "
                    f"{process.returncode}:\n{process.stdout}")
  records = []
  # Without stats, report.find_problems reports the simulator.
  if os.path.exists(stats_path):
    try:
      records = report.read_stats([stats_path])
    except (OSError, ValueError) as e:
      problems.append(f"cannot read runtime stats from {stats_path}: {e}")
  return records, problems


def _parse_args(argv):
  """Parses the command line arguments |argv|."""
  parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  parser.add_argument("--run-module",
                      required=True,
                      help="Path to iree-run-module.")
  parser.add_argument("--module", required=True, help="CoralNPU module.")
  parser.add_argument("--compiler-metrics",
                      action="append",
                      required=True,
                      help="Compiler metrics bundles of the module "
                      "(repeatable).")
  parser.add_argument("--simulator",
                      action="append",
                      default=[],
                      help="Simulator to run on (repeatable; default mpact).")
  parser.add_argument("--function", default="main", help="Function to run.")
  parser.add_argument("--input",
                      action="append",
                      default=[],
                      help="iree-run-module --input value (repeatable).")
  parser.add_argument("--flag",
                      action="append",
                      default=[],
                      help="iree-run-module flag, e.g. --device=coralnpu "
                      "(repeatable).")
  parser.add_argument("--output-dir",
                      default=os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR"),
                      help="Output directory (default: "
                      "$TEST_UNDECLARED_OUTPUTS_DIR).")
  args = parser.parse_args(argv)
  if not args.output_dir:
    parser.error("--output-dir is required outside of bazel test")
  return args


def main(argv=None):
  """Runs the module and writes the report; returns the exit status."""
  args = _parse_args(argv)
  simulators = list(dict.fromkeys(args.simulator or ["mpact"]))
  os.makedirs(args.output_dir, exist_ok=True)
  try:
    bundles = report.load_bundles(args.compiler_metrics)
  except (OSError, ValueError) as e:
    print(f"error: {e}", file=sys.stderr)
    return 2

  records = []
  problems = []
  ok_simulators = []
  for simulator in simulators:
    simulator_records, simulator_problems = run_simulator(args, simulator)
    records += simulator_records
    problems += simulator_problems
    if not simulator_problems:
      ok_simulators.append(simulator)
  try:
    merged = report.build_report(bundles, records)
  except ValueError as e:
    print(f"error: {e}", file=sys.stderr)
    return 2
  problems += report.find_problems(merged, ok_simulators)
  text = report.format_pretty(merged)
  with open(os.path.join(args.output_dir, "report.md"), "w") as f:
    f.write(text)
  with open(os.path.join(args.output_dir, "report.json"), "w") as f:
    json.dump(merged, f, indent=2)
    f.write("\n")
  sys.stdout.write(text)
  print(f"Wrote report.md and report.json to {args.output_dir}",
        file=sys.stderr)
  for problem in problems:
    print(f"error: {problem}", file=sys.stderr)
  return 1 if problems else 0


if __name__ == "__main__":
  sys.exit(main())
