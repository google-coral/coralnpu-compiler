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

"""Shared RV32 toolchain flags and rules for bare-metal CoralNPU targets."""

# For debugging, replace -Os with -O0 -g.
RV32_COPTS = "-march=rv32imf_zicsr_zifencei_zbb -mabi=ilp32 -mcmodel=medany -ffreestanding -fno-builtin -I. -Os -ffunction-sections -fdata-sections"

# Mirrors the link flags in compiler/Target/CoralNPULinkerTool.cpp.
RV32_LDFLAGS = "--build-id=none -nostdlib -static --no-undefined --gc-sections --no-warn-rwx-segments"

RV32_TOOLCHAIN_DIR = "$$(dirname $$(dirname $(execpath @rv32_toolchain//:bin/riscv32-unknown-elf-clang)))"

RV32_CLANG_FLAGS = "--gcc-toolchain=" + RV32_TOOLCHAIN_DIR

RV32_IREE_EXECUTABLE_LIBRARY_HDR = "@iree_core//runtime/src/iree/hal/local:executable_library.h"

RV32_IREE_INC = "$$(dirname $$(dirname $$(dirname $$(dirname $(execpath %s)))))" % RV32_IREE_EXECUTABLE_LIBRARY_HDR

def coralnpu_rv32_elf(
        name,
        src,
        out = None,
        hdrs = [],
        dispatch = False,
        linker_script = "//crt:coralnpu_tcm_ld",
        tags = [],
        **kwargs):
    """Compiles a bare-metal RV32 C source file and links a CoralNPU ELF.

    Args:
      name: Name of the genrule target.
      src: C source file to compile.
      out: Output ELF filename (defaults to `<name without _elf>.elf`).
      hdrs: Additional header targets needed in the sandbox.
      dispatch: If True, includes IREE's `executable_library.h` and links
        `libcoralnpu_iree` + `libcoralnpu_crt` like `CoralNPULinkerTool.cpp`;
        otherwise links `libcoralnpu_crt` alone.
      linker_script: Linker script target.
      tags: Bazel tags (`target=coralnpu` is added automatically).
      **kwargs: Forwarded to `native.genrule`.
    """
    if out == None:
        out = (name[:-4] if name.endswith("_elf") else name) + ".elf"

    archives = [
        "//crt:libcoralnpu_iree",
        "//crt:libcoralnpu_crt",
    ] if dispatch else [
        "//crt:libcoralnpu_crt",
    ]
    extra_srcs = [RV32_IREE_EXECUTABLE_LIBRARY_HDR] if dispatch else []
    extra_inc = (" -I" + RV32_IREE_INC) if dispatch else ""

    all_tags = list(tags)
    if "target=coralnpu" not in all_tags:
        all_tags.append("target=coralnpu")

    native.genrule(
        name = name,
        srcs = [
            src,
            linker_script,
            "@rv32_toolchain//:all_files",
            "@rv32_toolchain//:bin/riscv32-unknown-elf-clang",
            "@rv32_toolchain//:bin/riscv32-unknown-elf-ld",
        ] + hdrs + extra_srcs + archives,
        outs = [out],
        cmd = """
            {clang} {clang_flags} {copts}{extra_inc} -c $(execpath {src}) -o $(@D)/{name}.o
            {ld} -o $@ {ldflags} -T $(execpath {linker_script}) $(@D)/{name}.o \\
                --whole-archive {archive_paths} --no-whole-archive
        """.format(
            archive_paths = " ".join(["$(execpath %s)" % a for a in archives]),
            clang = "$(execpath @rv32_toolchain//:bin/riscv32-unknown-elf-clang)",
            clang_flags = RV32_CLANG_FLAGS,
            copts = RV32_COPTS,
            extra_inc = extra_inc,
            ld = "$(execpath @rv32_toolchain//:bin/riscv32-unknown-elf-ld)",
            ldflags = RV32_LDFLAGS,
            linker_script = linker_script,
            name = name,
            src = src,
        ),
        tags = all_tags,
        **kwargs
    )
