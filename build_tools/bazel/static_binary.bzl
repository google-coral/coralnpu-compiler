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

"""Starlark transition rule to build fully static standalone FPGA binaries.

Remote FPGA host machines frequently run older Linux distributions than the
build workstation (for example, older GLIBC and GLIBCXX symbol versions). This
rule wraps an executable target (such as @iree_core//tools:iree-run-module)
in an outgoing configuration transition so it links completely statically with
the FPGA and MPACT backends built in, without:
  - modifying third_party/iree target definitions,
  - passing global --linkopt=-static (which breaks shared-library targets), or
  - invalidating the Bazel cache for exec-configured compiler tools.
"""

def _static_fpga_transition_impl(settings, _attr):
    """Transitions a binary target to static linking with the FPGA backend."""

    # Preserve existing workspace linkopts (such as -fuse-ld=lld and
    # -Wl,--fatal-warnings from .bazelrc) and append static link flags:
    #   -static: Link libc, libm, libstdc++, and all runtime archives
    #       statically so the binary has zero runtime .so dependencies.
    #   -Wno-unused-command-line-argument: Silence Clang's warning that the
    #       toolchain's default -pie flag is unused when -static is active.
    #   -Wl,-S: Strip debug info sections to keep the archive compact while
    #       retaining symbol tables for stack traces.
    linkopts = list(settings["//command_line_option:linkopt"])
    for flag in ["-static", "-Wno-unused-command-line-argument", "-Wl,-S"]:
        if flag not in linkopts:
            linkopts.append(flag)

    return {
        "//command_line_option:linkopt": linkopts,
        # Link //runtime/sim/fpga:simulator directly into the HAL driver
        # registration module and disable dlopen()-based simulator loading
        # (calling dlopen from a -static glibc binary crashes in TLS setup).
        "//runtime/driver/registration:static_fpga": True,
    }

_static_fpga_transition = transition(
    implementation = _static_fpga_transition_impl,
    inputs = ["//command_line_option:linkopt"],
    outputs = [
        "//command_line_option:linkopt",
        "//runtime/driver/registration:static_fpga",
    ],
)

def _static_fpga_binary_impl(ctx):
    """Copies the transitioned static executable to a package-local output."""

    # Bazel wraps 1:1 transitioned label attributes in a single-element list.
    target = ctx.attr.target[0] if type(ctx.attr.target) == "list" else ctx.attr.target
    src_exe = target[DefaultInfo].files_to_run.executable
    out = ctx.actions.declare_file(ctx.attr.out or ctx.label.name)

    # Copy rather than symlink so downstream pkg_tar rules archive the regular
    # executable file contents instead of a symlink into the transitioned root.
    ctx.actions.run_shell(
        inputs = [src_exe],
        outputs = [out],
        arguments = [src_exe.path, out.path],
        command = 'cp -f "$1" "$2" && chmod +x "$2"',
        mnemonic = "CopyStaticFpgaBinary",
    )
    return [
        DefaultInfo(
            executable = out,
            files = depset([out]),
        ),
    ]

static_fpga_binary = rule(
    implementation = _static_fpga_binary_impl,
    doc = "Builds an executable target as a standalone static binary with the FPGA backend linked in.",
    attrs = {
        "out": attr.string(
            doc = "Output path relative to the package; defaults to the target name.",
        ),
        "target": attr.label(
            cfg = _static_fpga_transition,
            doc = "Executable target to rebuild in the static FPGA configuration.",
            executable = True,
            mandatory = True,
        ),
        "_allowlist_function_transition": attr.label(
            default = "@bazel_tools//tools/allowlists/function_transition_allowlist",
        ),
    },
    executable = True,
)
