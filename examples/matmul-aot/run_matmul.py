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
import sys
import time
import numpy as np
import iree.runtime as ireert


def init_iree_func(vmfb_path):
  instance = ireert.VmInstance()

  print("Available drivers:", ireert.query_available_drivers())

  try:
    cpu_device = ireert.get_device("local-sync")
    print("Created CPU device")
  except Exception as e:
    print(f"Failed to create CPU device: {e}")
    raise e

  try:
    npu_device = ireert.get_device("coralnpu")
    print("Created NPU device")
  except Exception as e:
    print(f"Failed to create NPU device: {e}")
    raise e

  # Create HAL module with both devices
  hal_module = ireert.create_hal_module(instance,
                                        devices=[cpu_device, npu_device])

  # Duck-typed config for SystemContext
  class MultiDeviceConfig:

    def __init__(self, device, instance, hal_module):
      self.device = device  # Used by FunctionInvoker for arguments
      self.vm_instance = instance
      self.default_vm_modules = (hal_module,)

  config = MultiDeviceConfig(cpu_device, instance, hal_module)

  print(f"Loading VMFB from {vmfb_path}...")
  try:
    vm_module = ireert.VmModule.mmap(instance, vmfb_path)
    print("Successfully mmapped VMFB")
  except Exception as e:
    print(f"mmap failed: {e}. Trying from_flatbuffer...")
    with open(vmfb_path, "rb") as f:
      vm_module = ireert.VmModule.from_flatbuffer(instance, f.read())
      print("Successfully loaded VMFB from flatbuffer")

  print("Creating SystemContext...")
  ctx = ireert.SystemContext(config=config)
  print("Successfully created SystemContext")

  print("Adding VM module to context...")
  ctx.add_vm_module(vm_module)
  print("Successfully added VM module")

  print("Resolving main function...")
  main_func = ctx.modules.jit_predict.main
  print("Successfully resolved main function")
  return main_func


def main():
  parser = argparse.ArgumentParser(description="Run matmul VMFB on CoralNPU")
  parser.add_argument("--vmfb", default=None, help="Path to VMFB file")
  args, _ = parser.parse_known_args()

  vmfb_path = args.vmfb if args.vmfb else "./matmul.vmfb"
  # If running via Bazel, we might need to find it relative to script
  if not os.path.exists(vmfb_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    vmfb_path = os.path.join(script_dir, os.path.basename(vmfb_path))

  if not os.path.exists(vmfb_path):
    print(f"Error: VMFB file {vmfb_path} not found. Please compile it first.")
    sys.exit(1)

  print("Initializing IREE...")
  predict_func = init_iree_func(vmfb_path)
  print("IREE initialized.")

  # Generate random inputs
  np.random.seed(42)
  x = np.random.randn(128, 128).astype(np.float32)
  y = np.random.randn(128, 128).astype(np.float32)

  print("Running inference...")
  t0 = time.time()
  output = predict_func(x, y)
  elapsed = time.time() - t0
  print(f"Inference completed in {elapsed:.4f}s")

  output_np = np.asarray(output)
  expected = x @ y

  print("Verifying results...")
  if np.allclose(output_np, expected, atol=1e-4, rtol=1e-4):
    print("SUCCESS: Results match numpy reference!")
  else:
    print("ERROR: Results mismatch!")
    print("Max diff:", np.max(np.abs(output_np - expected)))
    sys.exit(1)


if __name__ == "__main__":
  main()
