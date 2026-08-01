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

"""Bazel rule for compiling CoralNPU register allocation reports from MLIR files."""

def _coralnpu_register_allocation_report_impl(ctx):
    compile_tool = ctx.executable.compile_tool
    default_flags = [
        "--iree-hal-target-device=cpu=local",
        "--iree-hal-local-target-device-backends=llvm-cpu",
        "--iree-hal-default-device=cpu",
        "--iree-hal-target-device=npu=coralnpu",
        "--coralnpu-target-abi=ilp32",
        "--coralnpu-target-cpu-features=+m,+f,+zvl128b,+zve32f",
        "--iree-global-opt-experimental-disable-conv-generalization",
        "--coralnpu-dump-register-allocation-report-format=json",
        "--iree-llvmcpu-embedded-linker-path=" + ctx.executable.linker_tool.path,
        "--mlir-disable-threading",
    ]
    flags = default_flags + ctx.attr.flags

    outputs = []
    for src in ctx.files.srcs:
        out_file = ctx.actions.declare_file(ctx.label.name + "/" + src.basename + ".regalloc.json")
        outputs.append(out_file)

        tmp_dir = out_file.path + ".tmp"
        args = ctx.actions.args()
        for flag in flags:
            args.add(flag)
        args.add("--coralnpu-dump-register-allocation-report-dir=" + tmp_dir)
        args.add(src)
        args.add("-o", "/dev/null")

        cmd = " && ".join([
            "mkdir -p {tmp_dir}",
            'if ! {compile_tool} "$@" > /dev/null 2>&1; then echo \'{{"failed_to_compile": true}}\' > {out_file}; rm -rf {tmp_dir}; exit 0; fi',
            "shopt -s nullglob",
            "set -- {tmp_dir}/*_regalloc.json",
            'if [ "$#" -gt 1 ]; then echo "Error: Expected at most 1 *_regalloc.json file for {src}, found $#" >&2; exit 1; fi',
            'if [ "$#" -eq 1 ]; then mv "$1" {out_file}; else echo \'{{"dispatches": []}}\' > {out_file}; fi',
            "rm -rf {tmp_dir}",
        ]).format(
            tmp_dir = tmp_dir,
            compile_tool = compile_tool.path,
            out_file = out_file.path,
            src = src.basename,
        )

        ctx.actions.run_shell(
            outputs = [out_file],
            inputs = [src],
            tools = [compile_tool, ctx.executable.linker_tool],
            command = cmd,
            arguments = [args],
            mnemonic = "RegAllocReport",
            progress_message = "Generating CoralNPU register allocation report for %s" % src.basename,
        )

    return [DefaultInfo(
        files = depset(outputs),
    )]

coralnpu_register_allocation_report = rule(
    implementation = _coralnpu_register_allocation_report_impl,
    attrs = {
        "srcs": attr.label_list(
            allow_files = [".mlir"],
            mandatory = True,
            doc = "MLIR input files to compile.",
        ),
        "compile_tool": attr.label(
            default = Label("//compiler/tools:coralnpu-compile"),
            executable = True,
            cfg = "exec",
            doc = "Compiler binary to use.",
        ),
        "linker_tool": attr.label(
            default = Label("@llvm-project//lld:lld"),
            executable = True,
            cfg = "exec",
        ),
        "flags": attr.string_list(
            default = [],
            doc = "Additional compiler flags.",
        ),
    },
)
