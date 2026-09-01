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

import jax
import jax.numpy as jnp


def main():
  parser = argparse.ArgumentParser(
      description="Export matmul model to StableHLO MLIR")
  parser.add_argument(
      "-n",
      "--size",
      type=int,
      default=128,
      dest="n",
      help="Matrix dimension N for NxN matmul (default: 128)",
  )
  parser.add_argument("--output",
                      required=True,
                      help="Path to output MLIR file")
  parser.add_argument(
      "--transpose-lhs",
      action="store_true",
      help="Transpose LHS input before matmul (x.T @ y)",
  )
  parser.add_argument(
      "--int8",
      action="store_true",
      help="Export INT8 inputs with INT32 accumulator",
  )
  args = parser.parse_args()

  # This name becomes the VM module name ("jit_predict"); run_matmul.py
  # looks it up by that name.
  @jax.jit
  def predict(x, y):
    lhs = x.T if args.transpose_lhs else x
    if args.int8:
      return jnp.matmul(lhs, y, preferred_element_type=jnp.int32)
    return lhs @ y

  dtype = jnp.int8 if args.int8 else jnp.float32
  lhs_dummy = jnp.zeros([args.n, args.n], dtype=dtype)
  rhs_dummy = jnp.zeros([args.n, args.n], dtype=dtype)

  lowered = predict.lower(lhs_dummy, rhs_dummy)
  stablehlo_ir = lowered.compiler_ir(dialect="stablehlo")

  print(f"Writing MLIR to {args.output}...")
  os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
  with open(args.output, "w") as f:
    f.write(str(stablehlo_ir))


if __name__ == "__main__":
  main()
