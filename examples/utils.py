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
"""Shared utilities for CoralNPU AOT examples."""

import argparse
import os
import sys

import iree.runtime as ireert
import jax
import jax.numpy as jnp
import numpy as np

if "TEST_TMPDIR" in os.environ:
  os.environ["HOME"] = os.environ["TEST_TMPDIR"]


def parse_export_args():
  """Parses standard CLI arguments for CoralNPU example export scripts."""
  parser = argparse.ArgumentParser()
  parser.add_argument("--output",
                      required=True,
                      help="Path to output .mlir file")
  return parser.parse_args()


def export_stablehlo(model_fn, input_shape, output_path, dtype=jnp.float32):
  """Lowers a JAX-traceable callable to StableHLO MLIR and writes it to disk."""

  @jax.jit
  def predict(x):
    return model_fn(x)

  dummy_input = jnp.zeros(input_shape, dtype=dtype)
  stablehlo_ir = predict.lower(dummy_input).compiler_ir(dialect="stablehlo")

  with open(output_path, "w") as f:
    f.write(str(stablehlo_ir))
  print(f"Exported {output_path}")


def load_vmfb(vmfb_path, module_name="jit_predict"):
  """Loads a compiled VMFB onto the local-sync CPU + coralnpu NPU devices."""
  if not os.path.exists(vmfb_path):
    print(f"Error: VMFB file {vmfb_path} not found. Please compile it first.")
    sys.exit(1)

  instance = ireert.VmInstance()
  cpu_device = ireert.get_device("local-sync")
  npu_device = ireert.get_device("coralnpu")
  hal_module = ireert.create_hal_module(instance,
                                        devices=[cpu_device, npu_device])

  class MultiDeviceConfig:

    def __init__(self, device, instance, hal_module):
      self.device = device
      self.vm_instance = instance
      self.default_vm_modules = (hal_module,)

  config = MultiDeviceConfig(cpu_device, instance, hal_module)
  try:
    vm_module = ireert.VmModule.mmap(instance, vmfb_path)
  except PermissionError:
    with open(vmfb_path, "rb") as f:
      vm_module = ireert.VmModule.from_flatbuffer(instance, f.read())
  ctx = ireert.SystemContext(config=config)
  ctx.add_vm_module(vm_module)
  return getattr(ctx.modules, module_name).main


def parse_inference_args():
  """Parses standard CLI arguments for CoralNPU example inference scripts."""
  parser = argparse.ArgumentParser()
  parser.add_argument("input_path", nargs="?")
  parser.add_argument("--vmfb",
                      required=True,
                      help="Path to compiled .vmfb file")
  parser.add_argument(
      "--simulator",
      default="mpact",
      choices=["mpact", "verilator", "fpga"],
      help="Simulator backend to use",
  )
  parser.add_argument("--export-results", help="Path to save output .npy file")
  parser.add_argument("--check", help="Path to reference .npy file")
  args = parser.parse_args()
  ireert.flags.parse_flags(f"--simulator={args.simulator}")
  return args


def handle_results(results, args, rtol=1e-2, atol=1e-2):
  """Exports or checks computed inference results against a reference .npy."""
  if args.export_results:
    np.save(args.export_results, results)
    print(f"Saved results to {args.export_results}")
  if args.check:
    np.testing.assert_allclose(results,
                               np.load(args.check),
                               rtol=rtol,
                               atol=atol)
    print(f"Reference check passed ({args.check})")
