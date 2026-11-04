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
"""Workload report: merges the compiler metrics and the runtime stats.

Reads
  - compiler metrics bundles (--compiler-metrics, schema
    "coralnpu.metrics.v1", from iree-compile
    --coralnpu-dump-metrics-format=json): files with one or more bundles, or
    directories of *.metrics.json files;
  - runtime stats (--runtime-stats, JSON Lines of "coralnpu.stats.v1", from
    --coralnpu_stats_format=json),
joins the dispatches of the bundles with the stats records on (executable,
dispatch name), and reports, per dispatch and for the whole workload:
  - the cycles of the runs and of their kernels (the workgroup loops);
  - the ITCM and DTCM use of the firmware, the heap and the stack peak;
  - the tile grid of the root op;
  - the static Zvt lowering rate: the Zvt candidates that the compiler lowered
    to Zvt.
Prints the report as markdown (--format=pretty) or as one JSON document
(--format=json, schema "coralnpu.report.v1"). Values that were not reported
are left out of the JSON and shown as n/a.

The JSON document has these keys:
  schema       "coralnpu.report.v1"
  target       the target of the first bundle (with itcm_kb from the ELF)
  simulators   [the backend.simulator values of the stats]
  executables  [{executable, itcm, dtcm}]: the ELF regions of the bundles
  dispatches   [{executable, name, in_bundle, itcm_used, data_used, tile_grid,
               zvt, runs}]: in_bundle is whether a bundle had the dispatch;
               tile_grid and zvt come from the bundle; itcm_used and data_used
               come from the bundle (or the runtime stats when absent); runs is
               {<simulator>: run}
  summary      {firmware, zvt, simulators}: firmware is {executables,
               itcm_used, dtcm_used}, the peak static ELF region use across
               executables; zvt is {candidates, lowered_candidates, rate}, the
               static Zvt lowering rate over the dispatches of the bundles;
               simulators is {<simulator>: {runs, failures, cycles, instret,
               memory}}, over the runs of all the dispatches
A run aggregates the stats records of one dispatch on one simulator:
  runs, failures   the number of records, and of the failed ones
  cycles           {total, kernel}: sums over the runs
  instret          {kernel}: sum over the runs
  cycles_per_run   {total, kernel}: means over the successful runs
  instret_per_run  {kernel}: the mean over the successful runs
  memory           {heap_used, stack_peak, stack_overflow}: maxima over the
                   runs
A sum or per-run mean is left out unless all the runs that it covers reported
the value.
Adding keys keeps the schema version; renaming or removing keys bumps it.

Usage:
  report.py --compiler-metrics model.metrics.json \\
      --runtime-stats stats.jsonl [--format=pretty] [--output FILE]
"""

import argparse
import json
import os
import sys

SCHEMA = "coralnpu.report.v1"
_BUNDLE_SCHEMA = "coralnpu.metrics.v1"
_STATS_SCHEMA = "coralnpu.stats.v1"


def load_bundles(paths):
  """Returns the compiler metrics bundles of |paths|.

  A path is a file with one or more JSON bundles, one after the other (e.g.
  the output of --coralnpu-dump-metrics-dir=-), or a directory of
  *.metrics.json files, read in name order. Raises OSError or ValueError if a
  file can't be read.
  """
  files = []
  for path in paths:
    if not os.path.isdir(path):
      files.append(path)
      continue
    files += [
        os.path.join(path, name)
        for name in sorted(os.listdir(path))
        if name.endswith(".metrics.json")
    ]
  seen = set()
  bundles = []
  decoder = json.JSONDecoder()
  for path in files:
    with open(path) as f:
      text = f.read().lstrip()
    # Decode the bundles one after the other, skipping the whitespace between
    # them.
    while text:
      bundle, end = decoder.raw_decode(text)
      text = text[end:].lstrip()
      if not isinstance(bundle, dict) or bundle.get("schema") != _BUNDLE_SCHEMA:
        continue
      executable = bundle.get("executable")
      if executable in seen:
        raise ValueError(f"duplicate compiler metrics bundle for {executable}")
      seen.add(executable)
      bundles.append(bundle)
  return bundles


def read_stats(paths):
  """Returns the runtime stats records of the JSON Lines files |paths|."""
  records = []
  for path in paths:
    with open(path) as f:
      for line in f:
        if not line.strip():
          continue
        record = json.loads(line)
        if isinstance(record, dict) and record.get("schema") == _STATS_SCHEMA:
          records.append(record)
  return records


def _merge_memory(memory, record_memory):
  """Folds the `memory` object of a stats record into the maxima |memory|."""
  stack = record_memory.get("stack", {})
  values = (
      ("heap_used", record_memory.get("heap", {}).get("used")),
      ("stack_peak", stack.get("peak")),
      ("stack_overflow", stack.get("overflow")),
  )
  for key, value in values:
    if value is not None:
      memory[key] = max(memory.get(key, value), value)


def _merge_runs(records):
  """Aggregates the stats records of one dispatch on one simulator."""
  run = {"runs": len(records), "failures": 0}
  # Sums over all runs (emitted only when every run reported the counter) and
  # over successful runs (for per-run means).
  sums = {}
  counts = {}
  ok_sums = {}
  ok_counts = {}
  memory = {}
  for record in records:
    ok = record.get("status", {}).get("code", "OK") == "OK"
    if not ok:
      run["failures"] += 1
    for key in ("cycles", "instret"):
      for field, value in record.get(key, {}).items():
        sums[(key, field)] = sums.get((key, field), 0) + value
        counts[(key, field)] = counts.get((key, field), 0) + 1
        if ok:
          ok_sums[(key, field)] = ok_sums.get((key, field), 0) + value
          ok_counts[(key, field)] = ok_counts.get((key, field), 0) + 1
    _merge_memory(memory, record.get("memory", {}))
  ok_runs = len(records) - run["failures"]
  for (key, field), total in sorted(ok_sums.items()):
    if ok_counts[(key, field)] == ok_runs:
      run.setdefault(f"{key}_per_run", {})[field] = total // ok_runs
  for (key, field), total in sorted(sums.items()):
    # Failed runs may lack the kernel counters: then there is no sum.
    if counts[(key, field)] == len(records):
      run.setdefault(key, {})[field] = total
  if memory:
    run["memory"] = memory
  return run


def _firmware_summary(executables):
  """Returns the peak static ITCM and DTCM usage across |executables|."""
  summary = {"executables": len(executables)}
  for region in ("itcm", "dtcm"):
    used = [
        exe[region]["used"]
        for exe in executables
        if "used" in exe.get(region, {})
    ]
    if used:
      summary[f"{region}_used"] = max(used)
  return summary


def _static_zvt(dispatches):
  """Returns the static Zvt lowering rate of the bundle |dispatches|."""
  zvt = [dispatch["zvt"] for dispatch in dispatches if "zvt" in dispatch]
  candidates = sum(1 for entry in zvt if entry.get("candidate"))
  lowered = sum(
      1 for entry in zvt if entry.get("candidate") and entry.get("lowered"))
  summary = {"candidates": candidates, "lowered_candidates": lowered}
  if candidates:
    summary["rate"] = round(lowered / candidates, 4)
  return summary


def _simulator_summary(dispatches, simulator):
  """Returns the summary of the runs of all |dispatches| on |simulator|."""
  runs = [
      dispatch["runs"][simulator]
      for dispatch in dispatches
      if simulator in dispatch.get("runs", {})
  ]
  summary = {
      "runs": sum(run["runs"] for run in runs),
      "failures": sum(run["failures"] for run in runs),
  }
  for key, fields in (("cycles", ("total", "kernel")), ("instret",
                                                        ("kernel",))):
    for field in fields:
      # Sum across dispatches only when every dispatch that ran has a complete sum.
      if runs and all(field in run.get(key, {}) for run in runs):
        summary.setdefault(key,
                           {})[field] = sum(run[key][field] for run in runs)
  memory = {}
  for run in runs:
    for key, value in run.get("memory", {}).items():
      memory[key] = max(memory.get(key, value), value)
  if memory:
    summary["memory"] = memory
  return summary


def build_report(bundles, records):
  """Merges the bundles and the stats records into a report document."""
  report = {"schema": SCHEMA}
  if bundles:
    report["target"] = dict(bundles[0].get("target", {}))
  # The records by dispatch and simulator, and the simulators in order.
  by_dispatch = {}
  simulators = []
  for record in records:
    simulator = record.get("backend", {}).get("simulator", "unknown")
    if simulator not in simulators:
      simulators.append(simulator)
    dispatch = record.get("dispatch", {})
    key = (dispatch.get("executable"), dispatch.get("name"))
    by_dispatch.setdefault(key, {}).setdefault(simulator, []).append(record)
  report["simulators"] = simulators

  report["executables"] = []
  dispatches = {}
  for bundle in bundles:
    exe_name = bundle.get("executable")
    regions = bundle.get("elf", {}).get("regions", {})
    report["executables"].append({"executable": exe_name, **regions})
    static_mem = {
        key: value for key, value in (
            ("itcm_used", regions.get("itcm", {}).get("used")),
            ("data_used", regions.get("dtcm", {}).get("used")),
        ) if value is not None
    }
    for compiler in bundle.get("dispatches", []):
      key = (exe_name, compiler["name"])
      dispatches[key] = {
          "executable": exe_name,
          "name": compiler["name"],
          "in_bundle": True,
          **static_mem,
          **{
              key: compiler[key] for key in ("tile_grid", "zvt") if key in compiler
          },
      }
  # Dispatches that ran without a bundle are appended at the end.
  for (exe_name, name), runs in by_dispatch.items():
    entry = dispatches.setdefault(
        (exe_name, name),
        {
            "executable": exe_name,
            "name": name,
            "in_bundle": False
        },
    )
    for simulator_records in runs.values():
      for record in simulator_records:
        mem = record.get("memory", {})
        for key, region in (("itcm_used", "itcm"), ("data_used", "dtcm")):
          used = mem.get(region, {}).get("used")
          if used is None:
            continue
          if entry.setdefault(key, used) != used:
            raise ValueError(
                f"{exe_name}::{name} {key} mismatch: {entry[key]} vs {used}")
    entry["runs"] = {
        simulator: _merge_runs(simulator_records)
        for simulator, simulator_records in runs.items()
    }
  report["dispatches"] = list(dispatches.values())
  itcm_kb = next((exe["itcm"]["size"] // 1024
                  for exe in report["executables"]
                  if "size" in exe.get("itcm", {})), None)
  if itcm_kb is not None and "target" in report:
    report["target"].setdefault("itcm_kb", itcm_kb)

  report["summary"] = {
      "firmware": _firmware_summary(report["executables"]),
      "zvt": _static_zvt(report["dispatches"]),
      "simulators": {
          simulator: _simulator_summary(report["dispatches"], simulator)
          for simulator in simulators
      },
  }
  return report


def find_problems(report, simulators):
  """Returns the problems of the report document |report|, as messages.

  The problems are missing data (no bundle, no stats from one of
  |simulators|, a dispatch that ran without a bundle), failed runs, and stack
  overflows.
  """
  problems = []
  if not report["executables"]:
    problems.append("no compiler metrics bundle")
  for simulator in simulators:
    if simulator not in report["simulators"]:
      problems.append(f"no runtime stats from {simulator}")
  for dispatch in report["dispatches"]:
    name = f"{dispatch['executable']}::{dispatch['name']}"
    if not dispatch.get("in_bundle"):
      problems.append(f"no compiler metrics for dispatch {name}")
    for simulator, run in dispatch.get("runs", {}).items():
      if run["failures"]:
        problems.append(f"{name} failed {run['failures']} of {run['runs']} "
                        f"runs on {simulator}")
      if run.get("memory", {}).get("stack_overflow"):
        problems.append(f"{name} overflowed the stack on {simulator}")
  return problems


def _fmt(value):
  """Formats |value| for a table cell; None is shown as n/a."""
  if value is None:
    return "n/a"
  if isinstance(value, bool):
    return "yes" if value else "no"
  if isinstance(value, list):
    return "x".join(str(item) for item in value)
  return str(value)


def _percent(rate):
  """Formats the rate |rate| as a percentage; None is shown as n/a."""
  return "n/a" if rate is None else f"{rate * 100:.1f}%"


def _table(header, rows):
  """Returns a space-aligned text table in a fenced code block."""
  cells = [header] + [[_fmt(cell) for cell in row] for row in rows]
  widths = [max(len(row[i]) for row in cells) for i in range(len(header))]
  lines = [
      "  ".join(cell.ljust(width)
                for cell, width in zip(row, widths)).rstrip()
      for row in cells
  ]
  return "```\n" + "\n".join(lines) + "\n```\n"


def _summary_lines(report):
  """Returns the lines of the workload summary of the pretty format."""
  target = report.get("target", {})
  lines = [
      f"- Target: {_fmt(target.get('triple'))}, VLEN "
      f"{_fmt(target.get('vlen'))}, Zvt {_fmt(target.get('zvt'))}, "
      f"ITCM {_fmt(target.get('itcm_kb'))} KB, "
      f"DTCM {_fmt(target.get('dtcm_kb'))} KB\n"
  ]
  firmware = report["summary"]["firmware"]
  lines.append(f"- Firmware: up to {_fmt(firmware.get('itcm_used'))} "
               f"bytes of ITCM and {_fmt(firmware.get('dtcm_used'))} "
               f"bytes of DTCM, without the heap and the stack, in "
               f"{firmware['executables']} executables\n")
  zvt = report["summary"]["zvt"]
  lines.append(f"- Static Zvt rate: {zvt['lowered_candidates']} of "
               f"{zvt['candidates']} Zvt candidates lowered to Zvt "
               f"({_percent(zvt.get('rate'))})\n")
  for simulator, summary in report["summary"]["simulators"].items():
    cycles = summary.get("cycles", {})
    lines.append(f"- {simulator}: {summary['runs']} dispatch runs, "
                 f"{summary['failures']} failed, {_fmt(cycles.get('total'))} "
                 f"cycles, {_fmt(cycles.get('kernel'))} in the kernels\n")
  return lines


def _fmt_zvt(zvt):
  """Formats the Zvt {candidate, lowered} dict for the `zvt` column."""
  if not zvt:
    return None
  candidate = _fmt(zvt.get("candidate"))
  lowered = _fmt(zvt.get("lowered"))
  return candidate if candidate == lowered else f"{candidate}/{lowered}"


def _dispatch_names(dispatches):
  """Returns the display name of each dispatch, qualified when non-unique."""
  counts = {}
  for dispatch in dispatches:
    name = dispatch["name"]
    counts[name] = counts.get(name, 0) + 1
  return {
      (dispatch["executable"], dispatch["name"]): (
          f"{dispatch['executable']}::{dispatch['name']}"
          if counts[dispatch["name"]] > 1 else dispatch["name"]
      ) for dispatch in dispatches
  }


def _dispatch_rows(report):
  """Returns the rows of the dispatch table: one per dispatch and simulator."""
  names = _dispatch_names(report["dispatches"])
  rows = []
  for dispatch in report["dispatches"]:
    name = names[(dispatch["executable"], dispatch["name"])]
    zvt = _fmt_zvt(dispatch.get("zvt"))
    # Emit one n/a row for a bundled dispatch that did not run on any simulator.
    runs = dispatch.get("runs") or {None: {}}
    for simulator, run in runs.items():
      cycles = run.get("cycles_per_run", {})
      memory = run.get("memory", {})
      stack = memory.get("stack_peak")
      if stack is not None and memory.get("stack_overflow"):
        stack = f"{stack}!"
      rows.append([
          name,
          simulator,
          run.get("runs"),
          run.get("failures"),
          cycles.get("total"),
          cycles.get("kernel"),
          run.get("instret_per_run", {}).get("kernel"),
          dispatch.get("itcm_used"),
          dispatch.get("data_used"),
          memory.get("heap_used"),
          stack,
          zvt,
      ])
  return rows


def _tile_grid_rows(report):
  """Returns the rows of the tile grid table: one per dispatch with a tile_grid."""
  names = _dispatch_names(report["dispatches"])
  rows = []
  for dispatch in report["dispatches"]:
    grid = dispatch.get("tile_grid")
    if grid is None:
      continue
    rows.append([
        names[(dispatch["executable"], dispatch["name"])],
        grid.get("op"),
        grid.get("loops"),
        grid.get("workgroup"),
        grid.get("dtcm"),
        grid.get("vector"),
        grid.get("zvt_grid") or "",
    ])
  return rows


def format_pretty(report):
  """Returns the report document |report| as markdown."""
  out = ["# CoralNPU workload report\n\n"]
  out += _summary_lines(report)
  out.append("\n## Dispatches\n\n")
  out.append(
      "Per run: the cycles of the run and of the kernel (the workgroup "
      "loop), the instructions of the kernel, the ITCM and static DTCM "
      "data of the executable, the heap and the stack peak in bytes "
      "(\"!\" for an overflow), and whether the root op is a Zvt "
      "candidate and lowered to Zvt (\"cand/low\" when they differ).\n\n")
  out.append(
      _table([
          "dispatch", "simulator", "runs", "failed", "cycles", "kernel",
          "kinstret", "itcm", "data", "heap", "stack", "zvt"
      ], _dispatch_rows(report)))
  rows = _tile_grid_rows(report)
  if rows:
    out.append("\n## Tile grid\n\n")
    out.append("The loop ranges of the root op, and its workgroup, DTCM and "
               "vector tile sizes; the Zvt grid is the number of 16x16 Zvt "
               "tiles of a vector tile along M and N.\n\n")
    out.append(
        _table(
            ["dispatch", "op", "loops", "workgroup", "dtcm", "vector", "zvt"],
            rows))
  return "".join(out)


def _parse_args(argv):
  """Parses the command line arguments |argv|."""
  parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  parser.add_argument("--compiler-metrics",
                      action="append",
                      default=[],
                      help="Compiler metrics bundles: a file, or a directory "
                      "of *.metrics.json files (repeatable).")
  parser.add_argument("--runtime-stats",
                      action="append",
                      default=[],
                      help="Runtime stats JSON Lines file (repeatable).")
  parser.add_argument("--format",
                      choices=("pretty", "json"),
                      default="pretty",
                      help="Output format.")
  parser.add_argument("--output", help="Output file (default: stdout).")
  return parser.parse_args(argv)


def main(argv=None):
  """Writes the report of the inputs; returns 2 if one can't be read."""
  args = _parse_args(argv)
  try:
    bundles = load_bundles(args.compiler_metrics)
    records = read_stats(args.runtime_stats)
    report = build_report(bundles, records)
  except (OSError, ValueError) as e:
    print(f"error: {e}", file=sys.stderr)
    return 2
  if args.format == "json":
    text = json.dumps(report, indent=2) + "\n"
  else:
    text = format_pretty(report)
  if args.output:
    with open(args.output, "w") as f:
      f.write(text)
  else:
    sys.stdout.write(text)
  return 0


if __name__ == "__main__":
  sys.exit(main())
