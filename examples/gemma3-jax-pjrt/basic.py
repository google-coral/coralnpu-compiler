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

import sys
import jax
import jax.numpy as jnp
import numpy as np


def make_inputs():
  a = np.arange(1, 33, dtype=np.int32).reshape(4, 8)
  b = np.arange(1, 33, dtype=np.int32).reshape(8, 4)
  c = np.arange(1, 17, dtype=np.int32).reshape(4, 4)
  d = np.full((4, 4), 10, dtype=np.int32)
  return a, b, c, d


@jax.jit
def matmul_op(a, b):
  return jnp.matmul(a, b)


@jax.jit
def add_op(c, d):
  return c + d


def run_test(target):
  if isinstance(target, (list, tuple)):
    cpu_str, npu_str = str(target[0]), str(target[1])
    print(f"\n--- Test: Multi-Device ({cpu_str} + {npu_str}) ---")
    a, b, c, d = make_inputs()
    matmul_target, add_target = npu_str, cpu_str
  else:
    dev_name = str(target)
    print(f"\n--- Test: Single-Device ({dev_name}) ---")
    a, b, c, d = [jax.device_put(x, target) for x in make_inputs()]
    matmul_target = add_target = dev_name

  matmul_res = matmul_op(a, b)
  print(f"MatMul on {matmul_target} =")
  print(matmul_res)

  add_res = add_op(c, d)
  print(f"Add on {add_target} =")
  print(add_res)


def main():
  print(sys.executable)
  print(sys.version)
  print(f"jax version={jax.__version__}")

  jax.config.update("jax_platforms", "coralnpu_plugin")
  jax.config.update("jax_use_shardy_partitioner", False)

  devices = jax.devices("coralnpu_plugin")
  print(f"Devices found: {devices}")
  cpu_dev, npu_dev = devices[0], devices[1]
  print(f"Using Device 0: {cpu_dev}")
  print(f"Using Device 1: {npu_dev}")

  run_test(cpu_dev)
  run_test(npu_dev)
  run_test([cpu_dev, npu_dev])


if __name__ == "__main__":
  main()
