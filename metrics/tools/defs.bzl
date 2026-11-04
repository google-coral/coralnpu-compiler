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

"""Bazel macro for CoralNPU workload reports."""

load("@bazel_skylib//rules:native_binary.bzl", "native_test")
load("//build_tools/bazel:bytecode_module.bzl", "coralnpu_bytecode_module")

def coralnpu_workload_report(
        name,
        src,
        inputs = [],
        function = "main",
        simulators = ["mpact"],
        compiler_flags = [],
        run_module_flags = [],
        deps = [],
        timeout = "short",
        tags = []):
    """Reports the compiler metrics and the runtime stats of a workload.

    Compiles |src| with |compiler_flags| (<name>_module.vmfb), with its
    compiler metrics bundles as a second output (<name>.metrics.json). The
    test (metrics/tools/workload_report.py) runs |function| on |simulators|
    with iree-run-module, |run_module_flags| and the runtime stats, and merges
    the bundles and the stats with metrics/tools/report.py. It prints the
    report and writes report.md, report.json and the stats to its undeclared
    outputs (bazel-testlogs/<package>/<name>/test.outputs/). It fails if a
    run fails or if data is missing; never on performance.

    Args:
      name: Name of the test, and base name of the other targets.
      src: The MLIR source of the workload.
      inputs: The iree-run-module --input values of |function|, e.g.
        "4x8xf32=1".
      function: The function to run.
      simulators: The simulators to run on (default: ["mpact"]).
      compiler_flags: Extra iree-compile flags for the module.
      run_module_flags: Extra iree-run-module flags for the module.
      deps: Additional source files needed for compilation (e.g. linker
        scripts).
      timeout: Test timeout.
      tags: Tags for all the targets.
    """
    metrics = "%s.metrics.json" % name
    coralnpu_bytecode_module(
        name = "%s_module" % name,
        src = src,
        flags = ["--iree-hal-target-device=coralnpu"] + compiler_flags,
        metrics_name = metrics,
        deps = deps,
        tags = tags,
        visibility = ["//visibility:private"],
    )
    args = [
        "--run-module=$(rootpath @iree_core//tools:iree-run-module)",
        "--module=$(rootpath :%s_module.vmfb)" % name,
        "--compiler-metrics=$(rootpath :%s)" % metrics,
        "--function=%s" % function,
    ]
    args += ["--simulator=%s" % sim for sim in simulators]
    args += ["--flag=%s" % flag for flag in ["--device=coralnpu"] + run_module_flags]
    args += ["--input=%s" % value for value in inputs]

    data = [
        ":%s_module.vmfb" % name,
        ":%s" % metrics,
        "@iree_core//tools:iree-run-module",
    ]
    ld_paths = []
    for sim in simulators:
        if sim == "verilator":
            data.append("@coralnpu_hw//hw_sim:libcoralnpu_simulator_vme.so")
            ld_paths.append("../coralnpu_hw+/hw_sim:../coralnpu_hw/hw_sim:external/coralnpu_hw+/hw_sim:external/coralnpu_hw/hw_sim")
        elif sim in ("fpga", "spike"):
            data.append("//runtime/sim/%s:libcoralnpu_simulator_%s.so" % (sim, sim))
            ld_paths.append("runtime/sim/%s:../_main/runtime/sim/%s:../coralnpu-compiler/runtime/sim/%s" % (sim, sim, sim))

    default_tags = ["driver=coralnpu", "target=coralnpu"] + [
        "simulator=%s" % sim
        for sim in simulators
    ]
    native_test(
        name = name,
        src = "//metrics/tools:workload_report",
        args = args,
        data = data,
        env = {"LD_LIBRARY_PATH": ":".join(ld_paths)} if ld_paths else {},
        timeout = timeout,
        tags = tags + [
            tag
            for tag in default_tags
            if tag not in tags
        ],
    )
