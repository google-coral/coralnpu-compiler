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
import sys
import numpy as np
import iree.runtime as ireert

# jax.jit names the VM module after the function it wraps ("predict" ->
# "jit_predict") and the entry point is always "main".
VM_MODULE_NAME = "jit_predict"
VM_ENTRY_POINT = "main"


def init_iree_func(vmfb_path):
  instance = ireert.VmInstance()
  cpu_device = ireert.get_device("local-sync")
  npu_device = ireert.get_device("coralnpu")
  hal_module = ireert.create_hal_module(instance,
                                        devices=[cpu_device, npu_device])

  class MultiDeviceConfig:

    def __init__(self):
      self.device = cpu_device
      self.vm_instance = instance
      self.default_vm_modules = (hal_module,)

  # mmap is preferred, but it fails on filesystems that do not support it
  # (e.g. some network mounts); fall back to reading the whole flatbuffer.
  try:
    vm_module = ireert.VmModule.mmap(instance, vmfb_path)
  except Exception:  # pylint: disable=broad-except
    with open(vmfb_path, "rb") as f:
      vm_module = ireert.VmModule.from_flatbuffer(instance, f.read())

  ctx = ireert.SystemContext(config=MultiDeviceConfig())
  ctx.add_vm_module(vm_module)
  try:
    module = getattr(ctx.modules, VM_MODULE_NAME)
  except (AttributeError, KeyError) as e:
    raise RuntimeError(
        f"VMFB {vmfb_path} has no VM module named '{VM_MODULE_NAME}'; "
        "it is derived from the @jax.jit function name in export_matmul.py "
        "and the two must be kept in sync.") from e
  return getattr(module, VM_ENTRY_POINT)


def main():
  parser = argparse.ArgumentParser(description="Run matmul VMFB on CoralNPU")
  parser.add_argument(
      "-n",
      "--size",
      type=int,
      default=128,
      dest="n",
      help="Matrix dimension N for NxN matmul (default: 128)",
  )
  parser.add_argument("--vmfb", required=True, help="Path to VMFB file")
  parser.add_argument("--transpose-lhs",
                      action="store_true",
                      help="Expect x.T @ y")
  parser.add_argument("--int8",
                      action="store_true",
                      help="Use INT8 inputs and INT32 outputs")
  args = parser.parse_args()

  # The simulator backend is selected by the CORALNPU_SIMULATOR environment
  # variable, which the HAL driver reads at registration time.
  predict_func = init_iree_func(args.vmfb)

  # Generate random inputs
  np.random.seed(42)
  if args.int8:
    x = np.random.randint(-128, 128, size=(args.n, args.n), dtype=np.int8)
    y = np.random.randint(-128, 128, size=(args.n, args.n), dtype=np.int8)
  else:
    x = np.random.randn(args.n, args.n).astype(np.float32)
    y = np.random.randn(args.n, args.n).astype(np.float32)

  output = predict_func(x, y)

  output_np = np.asarray(output)
  lhs = x.T if args.transpose_lhs else x
  expected = (lhs.astype(np.int32) @ y.astype(np.int32)) if args.int8 else (
      lhs @ y)

  matches = (np.array_equal(output_np, expected) if args.int8 else np.allclose(
      output_np, expected, atol=1e-4, rtol=1e-4))
  if matches:
    print("SUCCESS: Results match numpy reference!")
  else:
    diff = np.abs(output_np - expected)
    print("ERROR: Results mismatch!")
    print("Max diff:", np.max(diff))
    sys.exit(1)


if __name__ == "__main__":
  main()
