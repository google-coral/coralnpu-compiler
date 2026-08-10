#!/usr/bin/env python3
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
"""Dynamic register usage benchmarking and spill analysis script for CoralNPU.

Analyzes generated JSON register allocation reports produced by Bazel rules,
aggregates multi-dimensional statistics, performs root-cause diagnostic
analysis of stack spills, and suggests targeted compiler fixes.
"""

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

# Register utilization classification statuses based on CoralNPU 32-reg budget.
#  - <8 regs: Under-Utilized (potential for aggressive unrolling/tiling)
#  - 8-16 regs: Moderate utilization
#  - 17-32 regs: Optimal utilization
STATUS_FAILED_TO_COMPILE = "⚠ FAILED TO COMPILE"
STATUS_NOT_VEC = "⦻ NOT-VECTORIZED"
STATUS_VEC_SPILLING = "◙ VEC-REG-SPILL"
STATUS_UNDER_UTILIZED = "◔ UNDER-UTILIZED"
STATUS_MODERATE = "◑ MODERATE"
STATUS_OPTIMAL = "● OPTIMAL"
STATUS_HOST_NOT_NPU = "◻ HOST (Not NPU)"


def parse_args():
  parser = argparse.ArgumentParser(
      description="Benchmark & analyze CoralNPU register allocation & spills.")
  parser.add_argument(
      "inputs",
      nargs="+",
      help="Report JSON files to analyze.",
  )
  parser.add_argument(
      "--output-dir",
      default="./agent-temp",
      help="Directory to store summary reports.",
  )
  parser.add_argument(
      "--disable-overall-status",
      action="store_true",
      help="Disable 'Overall Status Breakdown' section in the report.",
  )
  parser.add_argument(
      "--disable-by-element-type",
      action="store_true",
      help="Disable 'Breakdown by Element Data Type' section in the report.",
  )
  parser.add_argument(
      "--disable-by-size-category",
      action="store_true",
      help=
      "Disable 'Breakdown by Data / Shape Size Category' section in the report.",
  )
  parser.add_argument(
      "--disable-by-op-kind",
      action="store_true",
      help="Disable 'Breakdown by Operation Kind' section in the report.",
  )
  parser.add_argument(
      "--disable-failed-models",
      action="store_true",
      help="Disable 'Models that Failed to Compile' section in the report.",
  )
  parser.add_argument(
      "--disable-not-vectorized-models",
      action="store_true",
      help="Disable 'Not-Vectorized Models' section in the report.",
  )
  parser.add_argument(
      "--disable-spilling-dispatches",
      action="store_true",
      help="Disable 'Spilling Dispatches' section in the report.",
  )
  parser.add_argument(
      "--disable-host-models",
      action="store_true",
      help="Disable 'Models Fully Offloaded to Host CPU' section in the report.",
  )
  return parser.parse_args()


def extract_test_name(report_file):
  """Extracts test/model name from report file path by stripping suffixes."""
  name = report_file.name
  for suffix in (".regalloc.json", ".json", "_check.mlir", ".mlir"):
    if name.endswith(suffix):
      name = name[:-len(suffix)]
  return name


def discover_report_files(inputs):
  """Collects report JSON files from explicit command-line arguments."""
  report_files = []
  seen = set()
  for item in inputs:
    report_path = Path(item).resolve()
    if report_path.is_file() and report_path.name not in seen:
      seen.add(report_path.name)
      report_files.append(report_path)
  return report_files


def load_report_file(report_file):
  """Reads a compiled report JSON file and extracts metadata."""
  try:
    report_json = json.loads(report_file.read_text(errors="replace"))
  except Exception as err:
    print(
        f"Warning: JSON parse error in {report_file.name}: {err}",
        file=sys.stderr,
    )
    report_json = {"dispatches": []}

  mlir_name = extract_test_name(report_file)
  report_json["filename"] = mlir_name
  report_json["element_type"] = detect_element_type(mlir_name)
  report_json["op_kind"] = detect_op_kind(mlir_name)
  report_json["size_category"] = detect_shape_category(
      mlir_name, report_json["element_type"])
  return report_json


def detect_element_type(filename):
  """Extracts element type from filename using regex."""
  match = re.search(r"_([if][0-9]+)(_|$)", filename)
  if match:
    return match.group(1)
  return "unknown"


def detect_op_kind(filename):
  """Extracts Linalg operation kind from filename prefix."""
  match = re.search(r"_[if][0-9]+", filename)
  if match:
    return filename[:match.end()]
  return "other"


def detect_shape_category(filename, element_type):
  """Estimates shape size category (Registers <=512B, DTCM 512B-32KB, RAM >32KB)."""
  name = filename.removesuffix(".mlir").removesuffix("_check")
  match = re.search(r"_([\d_-]+)$", name)
  if not match:
    return "Unknown"

  shapes_str = match.group(1)
  total_elements = 0

  for shape_str in shapes_str.split("-"):
    dims = [int(d) for d in shape_str.split("_") if d]
    if dims:
      total_elements += math.prod(dims)

  if total_elements == 0:
    return "Unknown"

  bytes_per_element = 1
  if element_type != "unknown":
    try:
      bit_width = int(re.search(r"\d+", element_type).group())
      bytes_per_element = max(1, bit_width // 8)
    except (ValueError, AttributeError):
      pass

  total_bytes = total_elements * bytes_per_element

  if total_bytes <= 512:
    return "Registers (<=512B)"
  return "DTCM (512B-32KB)" if total_bytes <= 32768 else "RAM (>32KB)"


def get_or_create_group(group_dict, key):
  """Returns group metrics dictionary, initializing it safely if missing."""
  if key not in group_dict:
    group_dict[key] = {
        "total": 0,
        "vec_spills": 0,
        "scalar_spill_dispatches": 0,
        "sum_regs": 0,
        "status_counts": {
            STATUS_FAILED_TO_COMPILE: 0,
            STATUS_NOT_VEC: 0,
            STATUS_VEC_SPILLING: 0,
            STATUS_UNDER_UTILIZED: 0,
            STATUS_MODERATE: 0,
            STATUS_OPTIMAL: 0,
            STATUS_HOST_NOT_NPU: 0,
        },
    }
  return group_dict[key]


def analyze_results(results):
  """Aggregates register utilization statistics across all dispatches."""
  stats = {
      "total_dispatches": 0,
      "status_counts": {
          STATUS_FAILED_TO_COMPILE: 0,
          STATUS_NOT_VEC: 0,
          STATUS_VEC_SPILLING: 0,
          STATUS_UNDER_UTILIZED: 0,
          STATUS_MODERATE: 0,
          STATUS_OPTIMAL: 0,
          STATUS_HOST_NOT_NPU: 0,
      },
      "scalar_spill_dispatches": 0,
      "by_element_type": {},
      "by_op_kind": {},
      "by_size_category": {},
      "dispatches": [],
      "host_offloaded_models": [],
      "failed_models": [],
      "not_vectorized_models": [],
  }

  for report in results:
    if not report:
      continue

    if report.get("failed_to_compile", False):
      stats["failed_models"].append(report)
      report.setdefault("dispatches", []).append({
          "name": "compilation_failure",
          "is_vectorized": False,
          "is_failed_model": True,
      })

    if "dispatches" not in report:
      print(
          f"Error: Invalid report format in {report.get('filename', 'unknown')}. Expected 'dispatches'.",
          file=sys.stderr)
      sys.exit(1)

    elem_type = report["element_type"]
    op_kind = report["op_kind"]
    size_cat = report["size_category"]
    fname = report["filename"]

    if not report["dispatches"]:
      stats["host_offloaded_models"].append(report)
      report["dispatches"].append({
          "name": "host_dispatch",
          "is_vectorized": False,
          "is_host_model": True,
      })

    has_not_vec = False
    for dispatch in report["dispatches"]:
      stats["total_dispatches"] += 1
      vec_spills = dispatch.get("vec_spills", 0)
      vec_reloads = dispatch.get("vec_reloads", 0)
      has_vec_spills = (vec_spills > 0) or (vec_reloads > 0)
      has_scalar_spills = dispatch.get("has_scalar_spills", False)
      is_host = dispatch.get("is_host", False)

      global_regs = dispatch.get("global_vector_registers_count", 0)
      loop_regs = max(
          (loop.get("vector_registers_used_count", 0)
           for loop in dispatch.get("loops", [])),
          default=0,
      )
      max_regs = max(global_regs, loop_regs)
      is_vectorized = max_regs > 0

      if has_scalar_spills:
        stats["scalar_spill_dispatches"] += 1

      is_host_model = dispatch.get("is_host_model", False)
      is_failed_model = dispatch.get("is_failed_model", False)

      if is_failed_model:
        status = STATUS_FAILED_TO_COMPILE
      elif is_host_model:
        status = STATUS_HOST_NOT_NPU
      elif not is_vectorized:
        status = STATUS_NOT_VEC
        has_not_vec = True
      elif has_vec_spills:
        status = STATUS_VEC_SPILLING
      elif max_regs < 8:
        status = STATUS_UNDER_UTILIZED
      elif max_regs < 17:
        status = STATUS_MODERATE
      else:
        status = STATUS_OPTIMAL

      stats["status_counts"][status] += 1
      stats["dispatches"].append({
          "filename": fname,
          "dispatch_name": dispatch.get("name", "unknown"),
          "element_type": elem_type,
          "op_kind": op_kind,
          "size_category": size_cat,
          "vec_spills": vec_spills,
          "vec_reloads": vec_reloads,
          "has_scalar_spills": has_scalar_spills,
          "vector_regs_used": max_regs,
          "status": status,
          "is_vectorized": is_vectorized,
      })

      for group_dict, group_key in (
          (stats["by_element_type"], elem_type),
          (stats["by_op_kind"], op_kind),
          (stats["by_size_category"], size_cat),
      ):
        group_stats = get_or_create_group(group_dict, group_key)
        group_stats["total"] += 1
        group_stats["vec_spills"] += vec_spills + vec_reloads
        group_stats["scalar_spill_dispatches"] += 1 if has_scalar_spills else 0
        group_stats["status_counts"][status] += 1
        group_stats["sum_regs"] += max_regs

    if has_not_vec:
      stats["not_vectorized_models"].append(report)

  return stats


def markdown_table(headers, rows):
  """Formats a list of headers and rows into a Markdown table string."""
  lines = ["| " + " | ".join(headers) + " |"]
  lines.append("| " + " | ".join(":---" for _ in range(len(headers))) + " |")
  for row in rows:
    lines.append("| " + " | ".join(str(col) for col in row) + " |")
  return "\n".join(lines)


def generate_markdown_report(stats, args=None):
  """Generates formatted markdown report with root-cause diagnosis."""
  total = stats["total_dispatches"]
  lines = [
      "# CoralNPU Register Allocation & Spill Report\n",
      f"**Total Dispatches Analyzed**: {total}\n",
  ]

  if not (args and getattr(args, "disable_overall_status", False)):
    lines.append("## Overall Status Breakdown\n")
    status_descs = {
        STATUS_FAILED_TO_COMPILE: "Compilation failed",
        STATUS_VEC_SPILLING: "Vector register stack spills present",
        STATUS_UNDER_UTILIZED: "< 8 vector registers used",
        STATUS_MODERATE: "8-16 vector registers used",
        STATUS_OPTIMAL: "17-32 vector registers used",
        STATUS_NOT_VEC: "Not vectorized",
        STATUS_HOST_NOT_NPU: "Offloaded to Host",
    }

    status_rows = [(
        status_name,
        count,
        f"{(count / total * 100) if total > 0 else 0.0:.1f}%",
        status_descs.get(status_name, ""),
    ) for status_name, count in stats["status_counts"].items()]
    lines.append(
        markdown_table(["Status", "Count", "Percentage", "Description"],
                       status_rows))
    lines.append("")

    scalar_cnt = stats.get("scalar_spill_dispatches", 0)
    scalar_pct = (scalar_cnt / total * 100) if total > 0 else 0.0
    lines.append(f"**Total Dispatches with Scalar Spills**: {scalar_cnt}"
                 f" ({scalar_pct:.1f}%)\n")

  def add_group_section(title, group_data, sort_key=lambda x: x[0]):
    lines.append(f"## Breakdown by {title}\n")
    headers = [
        "Category",
        "Total",
        "⚠ Failed",
        "⦻ Not Vec",
        "◙ Vec Spill",
        "◔ Under",
        "◑ Mod",
        "● Optimal",
        "◻ Host",
        "Avg Regs",
        "Scalar Spills",
    ]
    rows = []
    for category_name, group_stats in sorted(group_data.items(), key=sort_key):
      avg_regs = (group_stats["sum_regs"] /
                  group_stats["total"]) if group_stats["total"] > 0 else 0.0
      status_counts_dict = group_stats["status_counts"]
      rows.append((
          f"**{category_name}**",
          group_stats["total"],
          status_counts_dict[STATUS_FAILED_TO_COMPILE],
          status_counts_dict[STATUS_NOT_VEC],
          status_counts_dict[STATUS_VEC_SPILLING],
          status_counts_dict[STATUS_UNDER_UTILIZED],
          status_counts_dict[STATUS_MODERATE],
          status_counts_dict[STATUS_OPTIMAL],
          status_counts_dict[STATUS_HOST_NOT_NPU],
          f"{avg_regs:.1f}",
          group_stats["scalar_spill_dispatches"],
      ))
    lines.append(markdown_table(headers, rows))
    lines.append("")

  def sort_key_elem(item):
    match = re.match(r"([a-zA-Z]+)(\d+)", item[0])
    if match:
      return (int(match.group(2)), match.group(1))
    return (0, item[0])

  if not (args and getattr(args, "disable_by_element_type", False)):
    add_group_section("Element Data Type",
                      stats["by_element_type"],
                      sort_key=sort_key_elem)

  def sort_key_size(item):
    if "Registers" in item[0]:
      return 0
    if "DTCM" in item[0]:
      return 1
    if "RAM" in item[0]:
      return 2
    return 3

  if not (args and getattr(args, "disable_by_size_category", False)):
    add_group_section("Data / Shape Size Category",
                      stats["by_size_category"],
                      sort_key=sort_key_size)

  if not (args and getattr(args, "disable_by_op_kind", False)):
    add_group_section("Operation Kind", stats["by_op_kind"])

  if stats.get("failed_models") and not (args and getattr(
      args, "disable_failed_models", False)):
    lines.append("## Models that Failed to Compile\n")
    failed_rows = [(
        f"`{model_info['filename']}`",
        model_info["op_kind"],
        model_info["element_type"],
        model_info["size_category"],
    ) for model_info in stats["failed_models"]]
    lines.append(
        markdown_table(
            [
                "File",
                "Op Kind",
                "Type",
                "Size Category",
            ],
            failed_rows,
        ))
    lines.append("")

  if stats.get("not_vectorized_models") and not (args and getattr(
      args, "disable_not_vectorized_models", False)):
    lines.append("## Not-Vectorized Models\n")
    not_vec_rows = [(
        f"`{model_info['filename']}`",
        model_info["op_kind"],
        model_info["element_type"],
        model_info["size_category"],
    ) for model_info in stats["not_vectorized_models"]]
    lines.append(
        markdown_table(
            [
                "File",
                "Op Kind",
                "Type",
                "Size Category",
            ],
            not_vec_rows,
        ))
    lines.append("")

  if not (args and getattr(args, "disable_spilling_dispatches", False)):
    lines.append("## Spilling Dispatches\n")
    spilling_ops = [
        dispatch_info for dispatch_info in stats["dispatches"]
        if dispatch_info["status"] == STATUS_VEC_SPILLING
    ]
    if not spilling_ops:
      lines.append("✓ **No spilling dispatches detected!**\n")
    else:
      spill_rows = [(
          f"`{dispatch_info['filename']}`",
          dispatch_info["op_kind"],
          dispatch_info["element_type"],
          dispatch_info["vec_spills"],
          dispatch_info["vec_reloads"],
          "Yes" if dispatch_info["has_scalar_spills"] else "No",
          "Yes" if dispatch_info["is_vectorized"] else "No",
      ) for dispatch_info in spilling_ops]
      lines.append(
          markdown_table(
              [
                  "File",
                  "Op Kind",
                  "Type",
                  "Vec Spills",
                  "Vec Reloads",
                  "Has Scalar Spills",
                  "Vectorized",
              ],
              spill_rows,
          ))
      lines.append("")

  if stats.get("host_offloaded_models") and not (args and getattr(
      args, "disable_host_models", False)):
    lines.append("## Models Fully Offloaded to Host CPU\n")
    host_rows = [(
        f"`{model_info['filename']}`",
        model_info["op_kind"],
        model_info["element_type"],
        model_info["size_category"],
    ) for model_info in stats["host_offloaded_models"]]
    lines.append(
        markdown_table(
            [
                "File",
                "Op Kind",
                "Type",
                "Size Category",
            ],
            host_rows,
        ))
    lines.append("")

  return "\n".join(lines)


def main():
  args = parse_args()
  output_dir = Path(args.output_dir).resolve()
  output_dir.mkdir(parents=True, exist_ok=True)

  report_files = discover_report_files(args.inputs)
  if not report_files:
    print(
        "No register allocation report files (*.json) found in provided"
        " inputs.",
        file=sys.stderr,
    )
    sys.exit(1)

  print(
      f"Found {len(report_files)} compiled register allocation JSON reports...")
  raw_results = []
  for current_file in report_files:
    try:
      raw_results.append(load_report_file(current_file))
    except Exception as err:
      print(f"Error processing {current_file.name}: {err}", file=sys.stderr)

  stats = analyze_results(raw_results)

  summary_json_path = output_dir / "register_benchmark_summary.json"
  summary_json_path.write_text(json.dumps(stats, indent=2))
  print(f"\nSaved summary JSON to {summary_json_path}")

  report_md = generate_markdown_report(stats, args)
  report_md_path = output_dir / "register_benchmark_report.md"
  report_md_path.write_text(report_md)
  print(f"Saved Markdown report to {report_md_path}\n")

  print(report_md)


if __name__ == "__main__":
  main()
