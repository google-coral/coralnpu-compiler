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

"""Bazel macro for end-to-end tests of CoralNPU AOT examples."""

load("@bazel_skylib//rules:native_binary.bzl", "native_test")
load("@bazel_skylib//rules:run_binary.bzl", "run_binary")
load("//build_tools/bazel:bytecode_module.bzl", "coralnpu_bytecode_module")

def coralnpu_example_test(
        name,
        export_tool,
        runner_tool,
        reference,
        simulator = "mpact",
        highmem = False,
        timeout = None,
        tags = []):
    """Exports an example model to MLIR, compiles it to VMFB, and verifies execution against a reference .npy file.

    Args:
      name: Base name of the test target.
      export_tool: Label of the example's export py_binary target.
      runner_tool: Label of the example's inference py_binary target.
      reference: Label of the reference .npy output file (e.g. "//examples/anomaly-detection-jax-aot:reference.npy").
      simulator: Target simulator backend ("mpact", "verilator", or "fpga").
      highmem: Whether to compile using the 1024 KB ITCM/DTCM highmem linker script.
      timeout: Optional Bazel test timeout.
      tags: Additional Bazel test tags.
    """
    mlir_filename = "%s.mlir" % name
    vmfb_filename = "%s.vmfb" % name

    run_binary(
        name = "%s_mlir" % name,
        outs = [mlir_filename],
        args = [
            "--output=$(location :%s)" % mlir_filename,
        ],
        tags = tags + ["requires-network"],
        tool = export_tool,
    )

    compile_flags = [
        "--iree-hal-target-device=local",
        "--iree-hal-local-target-device-backends=llvm-cpu",
        "--iree-llvmcpu-target-cpu=host",
        "--iree-hal-target-device=coralnpu",
        "--iree-global-opt-experimental-disable-conv-generalization",
    ]
    compile_deps = []
    if highmem:
        compile_flags.extend([
            "--coralnpu-dtcm-size-kb=1024",
            "--coralnpu-linker-script-path=$(location //crt:coralnpu_tcm_highmem_ld)",
        ])
        compile_deps.append("//crt:coralnpu_tcm_highmem_ld")

    coralnpu_bytecode_module(
        name = "%s_vmfb" % name,
        src = ":%s" % mlir_filename,
        module_name = vmfb_filename,
        flags = compile_flags,
        deps = compile_deps,
        tags = tags,
    )

    test_data = [
        ":%s" % vmfb_filename,
        reference,
    ]
    test_env = {}
    if simulator == "verilator":
        test_data.append("@coralnpu_hw//hw_sim:libcoralnpu_simulator_rvv.so")
        test_env["LD_LIBRARY_PATH"] = "../coralnpu_hw+/hw_sim:../coralnpu_hw/hw_sim:external/coralnpu_hw+/hw_sim:external/coralnpu_hw/hw_sim"

    native_test(
        name = name,
        src = runner_tool,
        args = [
            "--vmfb=$(location :%s)" % vmfb_filename,
            "--check=$(location %s)" % reference,
            "--simulator=%s" % simulator,
        ],
        data = test_data,
        env = test_env,
        timeout = timeout,
        tags = tags + [
            "driver=coralnpu",
            "requires-network",
            "simulator=%s" % simulator,
            "target=coralnpu",
        ],
    )
