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

"""Custom Bazel macro for compiling CoralNPU bytecode modules with sandbox setup."""

def coralnpu_bytecode_module(
        name,
        src,
        flags,
        module_name = None,
        compile_tool = "@iree_core//tools:iree-compile",
        linker_tool = "@llvm-project//lld:lld",
        deps = [],
        **kwargs):
    """Builds an IREE bytecode module with CoralNPU sandbox setup.

    This macro wraps a genrule to compile an MLIR file to a VMFB. It copies the
    necessary CRT files and adds the toolchain to the PATH before running the
    compiler, allowing the compiler to find the linker and linker script
    inside the Bazel sandbox.
    """

    if not module_name:
        module_name = "%s.vmfb" % (name)

    out_files = [module_name]

    actual_flags = list(flags) + [
        "--output-format=vm-bytecode",
        "--mlir-print-op-on-diagnostic=false",
    ]

    cmd = " && ".join([
        "RUNFILES_DIR=$$(pwd)/$(location %s).runfiles" % (compile_tool),
        "if [ -d $$RUNFILES_DIR ]; then mkdir -p crt && cp $$RUNFILES_DIR/_main/crt/coralnpu_tcm.ld crt/ && cp $$RUNFILES_DIR/_main/crt/*.a crt/ && ln -sf $$RUNFILES_DIR/+coralnpuc_extension+rv32_toolchain +coralnpuc_extension+rv32_toolchain; fi",
        " ".join([
            "$(location %s)" % (compile_tool),
            " ".join(actual_flags),
            "--iree-llvmcpu-embedded-linker-path=$(location %s)" % (linker_tool),
            "--iree-llvmcpu-wasm-linker-path=$(location %s)" % (linker_tool),
            "-o $(location %s)" % (module_name),
            "$(location %s)" % (src),
        ]),
    ])

    native.genrule(
        name = name,
        srcs = [src],
        outs = out_files,
        cmd = cmd,
        tools = [compile_tool, linker_tool],
        message = "Compiling IREE module %s..." % (name),
        output_to_bindir = 1,
        **kwargs
    )
