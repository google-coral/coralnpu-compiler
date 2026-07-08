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

"""Starlark transition rule to build packaging targets for multiple platforms in a single invocation."""

def _multi_platform_transition_impl(_settings, _attr):
    return {
        "linux_x86_64": {
            "//command_line_option:platforms": ["//build_tools/bazel/platforms:linux_x86_64"],
        },
        # TODO: Enable additional target platforms when C++ cross-compiler toolchains are registered in MODULE.bazel.
        # "linux_aarch64": {
        #     "//command_line_option:platforms": ["//build_tools/bazel/platforms:linux_aarch64"],
        # },
        # "macosx_x86_64": {
        #     "//command_line_option:platforms": ["//build_tools/bazel/platforms:macosx_x86_64"],
        # },
        # "macosx_arm64": {
        #     "//command_line_option:platforms": ["//build_tools/bazel/platforms:macosx_arm64"],
        # },
        # "windows_x86_64": {
        #     "//command_line_option:platforms": ["//build_tools/bazel/platforms:windows_x86_64"],
        # },
    }

multi_platform_transition = transition(
    implementation = _multi_platform_transition_impl,
    inputs = [],
    outputs = ["//command_line_option:platforms"],
)

def _multi_platform_package_impl(ctx):
    out_files = []

    # Map platform keys to wheel platform tags for cross-compiled wheels
    # wheel_platform_tags = {
    #     "linux_x86_64": "manylinux_2_28_x86_64",
    #     "linux_aarch64": "manylinux_2_28_aarch64",
    #     "macosx_x86_64": "macosx_10_15_x86_64",
    #     "macosx_arm64": "macosx_11_0_arm64",
    #     "windows_x86_64": "win_amd64",
    # }

    for platform_key, target in ctx.split_attr.target.items():
        for file in target.files.to_list():
            # Create a platform-tagged symlink/output file
            ext = file.extension
            base_name = file.basename
            if ext:
                stem = base_name[:-len(ext) - 1]
                tagged_name = ctx.actions.declare_file(stem + "-" + platform_key + "." + ext)
            else:
                tagged_name = ctx.actions.declare_file(base_name + "-" + platform_key)

            ctx.actions.symlink(
                output = tagged_name,
                target_file = file,
            )
            out_files.append(tagged_name)

    return [DefaultInfo(files = depset(out_files))]

multi_platform_package = rule(
    implementation = _multi_platform_package_impl,
    attrs = {
        "target": attr.label(
            cfg = multi_platform_transition,
            mandatory = True,
        ),
        "_allowlist_function_transition": attr.label(
            default = "@bazel_tools//tools/allowlists/function_transition_allowlist",
        ),
    },
)
