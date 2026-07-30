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

import argparse
import os
import time

import jax
import jax.numpy as jnp


def main():
  parser = argparse.ArgumentParser(
      description="Export matmul model to StableHLO MLIR")
  parser.add_argument("--output", default=None, help="Path to output MLIR file")
  args, _ = parser.parse_known_args()

  @jax.jit
  def predict(x, y):
    return x @ y

  lhs_dummy = jnp.zeros([128, 128], dtype=jnp.float32)
  rhs_dummy = jnp.zeros([128, 128], dtype=jnp.float32)

  # Warmup/trace
  print("Tracing model...")
  _ = predict(lhs_dummy, rhs_dummy)
  print("Model traced.")

  print("Lowering to StableHLO...")
  t0 = time.time()
  lowered = predict.lower(lhs_dummy, rhs_dummy)
  stablehlo_ir = lowered.compiler_ir(dialect="stablehlo")
  print(f"Lowered in {time.time() - t0:.2f}s")

  if args.output:
    output_path = args.output
  else:
    workspace_dir = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
    if workspace_dir:
      output_dir = os.path.join(workspace_dir, "examples", "matmul-aot")
    else:
      output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, "matmul.mlir")

  print(f"Writing MLIR to {output_path}...")
  os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
  with open(output_path, "w") as f:
    f.write(str(stablehlo_ir))
  print("Done.")


if __name__ == "__main__":
  main()
