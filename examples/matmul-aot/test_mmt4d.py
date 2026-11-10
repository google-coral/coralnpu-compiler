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
"""End-to-end simulator verification for linalg.mmt4d on CoralNPU Zvt."""

import argparse
import glob
import os
import re
import subprocess
import sys
import tempfile
import numpy as np

MLIR_TEMPLATE = """
func.func @main(%arg0: tensor<2x16x16x1x{in_type}>, %arg1: tensor<2x16x16x1x{in_type}>, %arg2: tensor<2x2x16x16x{acc_type}>) -> tensor<2x2x16x16x{acc_type}> {{
  %zero = arith.constant {zero} : {acc_type}
  %empty = tensor.empty() : tensor<2x2x16x16x{acc_type}>
  %fill = linalg.fill ins(%zero : {acc_type}) outs(%empty : tensor<2x2x16x16x{acc_type}>) -> tensor<2x2x16x16x{acc_type}>
  %0 = linalg.mmt4d
    ins(%arg0, %arg1 : tensor<2x16x16x1x{in_type}>, tensor<2x16x16x1x{in_type}>)
    outs({init} : tensor<2x2x16x16x{acc_type}>) -> tensor<2x2x16x16x{acc_type}>
  return %0 : tensor<2x2x16x16x{acc_type}>
}}
"""


def find_tools(build_dir, force_cmake=False, force_bazel=False):
  root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
  if build_dir:
    dirs = [build_dir]
  else:
    dirs = [] if force_bazel else [
        os.path.join(root, "build-out"),
        os.path.join(root, "build")
    ]
    if not force_cmake:
      dirs.append(os.path.join(root, "bazel-bin"))

  def find(d, name):
    return [
        p for sub in ("third_party/iree/tools", "tools", "bin",
                      "external/*/tools")
        for p in glob.glob(os.path.join(d, sub, name))
    ]

  for d in dirs:
    c_hits, r_hits = find(d, "iree-compile"), find(d, "iree-run-module")
    if c_hits and r_hits:
      print(f"Using tools from {d}")
      return c_hits[0], r_hits[0]

  sys.exit("Error: Could not find build directory containing tools.")


def verify_dtype(dtype, zero_init, compile_tool, run_tool):
  init = "zero-fill" if zero_init else "accumulator"
  print(f"\n=== Verifying linalg.mmt4d ({dtype.upper()}, {init} init) ===")
  # bf16 is generated as f32, truncated below
  is_f32 = dtype not in ("i8", "i16")
  in_dt = np.float32 if is_f32 else (np.int16 if dtype == "i16" else np.int8)
  acc_dt = np.float32 if is_f32 else np.int32
  t_acc = "f32" if is_f32 else "i32"

  np.random.seed(42)
  if is_f32:
    gen = lambda s: np.random.uniform(-1.0, 1.0, s)
  else:
    lim = 1000 if dtype == "i16" else 4
    gen = lambda s: np.random.randint(-lim, lim, s)
  lhs, rhs = [gen((2, 16, 16, 1)).astype(in_dt) for _ in range(2)]
  ref = [x.astype(acc_dt) for x in (lhs, rhs)]
  if dtype == "bf16":
    # Truncate to raw bf16 bits (numpy has no bf16 type).
    ref = [(x.view(np.uint32) & 0xFFFF0000).view(np.float32) for x in ref]
    lhs, rhs = [(x.view(np.uint32) >> 16).astype(np.uint16) for x in ref]
  acc = np.zeros((2, 2, 16, 16), dtype=acc_dt)
  if not zero_init:
    acc += gen(acc.shape).astype(acc_dt)
  expected = (acc + np.einsum("abcd,ebfd->aecf", *ref)).flatten()

  with tempfile.TemporaryDirectory(prefix=f"mmt4d_{dtype}_") as tmpdir:
    lhs_bin, rhs_bin, acc_bin = [
        os.path.join(tmpdir, f"{n}.bin") for n in ("lhs", "rhs", "acc")
    ]
    lhs.tofile(lhs_bin)
    rhs.tofile(rhs_bin)
    acc.tofile(acc_bin)

    mlir_file = os.path.join(tmpdir, "model.mlir")
    vmfb_file = os.path.join(tmpdir, "model.vmfb")
    with open(mlir_file, "w") as f:
      f.write(
          MLIR_TEMPLATE.format(in_type=dtype,
                               acc_type=t_acc,
                               zero="0.0" if is_f32 else "0",
                               init="%fill" if zero_init else "%arg2"))

    subprocess.run(
        [
            compile_tool,
            mlir_file,
            "-o",
            vmfb_file,
            "--iree-hal-target-device=coralnpu",
            "--mlir-disable-threading",
        ],
        check=True,
    )
    res = subprocess.run(
        [
            run_tool,
            f"--module={vmfb_file}",
            "--device=coralnpu",
            "--function=main",
            f"--input=2x16x16x1x{dtype}=@{lhs_bin}",
            f"--input=2x16x16x1x{dtype}=@{rhs_bin}",
            f"--input=2x2x16x16x{t_acc}=@{acc_bin}",
            "--output_max_element_count=20000",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

  match = re.search(r"=\s*\[(.*)\]", res.stdout, re.DOTALL)
  if not match:
    raise ValueError(f"Could not parse output:\n{res.stdout}")
  actual = np.array(
      [float(x) for x in re.sub(r"[\[\]]", " ", match.group(1)).split()])

  tol = 1e-4 if is_f32 else 0
  max_diff = float(np.max(np.abs(expected - actual)))
  print(f"Max difference: {max_diff}")
  if max_diff <= tol:
    print(f"SUCCESS: linalg.mmt4d outputs match ({dtype.upper()}, {init}).")
    return True
  print(f"FAILED: linalg.mmt4d mismatch ({dtype.upper()}, {init}).")
  return False


def main():
  parser = argparse.ArgumentParser(
      description="Verify Zvt Matrix linalg.mmt4d on CoralNPU simulator.")
  parser.add_argument("--build-dir", default=None)
  group = parser.add_mutually_exclusive_group()
  group.add_argument("--cmake", action="store_true")
  group.add_argument("--bazel", action="store_true")
  args = parser.parse_args()

  compile_tool, run_tool = find_tools(args.build_dir, args.cmake, args.bazel)
  # A list, not a generator: run every case even if one mismatches.
  ok = [
      verify_dtype(t, z, compile_tool, run_tool)
      for t in ("f32", "bf16", "i16", "i8")
      for z in (True, False)
  ]
  sys.exit(0 if all(ok) else 1)


if __name__ == "__main__":
  main()
