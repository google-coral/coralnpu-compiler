"""Bzlmod extension for CoralNPU Compiler repository rules."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:local.bzl", "new_local_repository")

def _coralnpuc_extension_impl(module_ctx):
    # Create llvm-raw only when coralnpuc is the root module.
    # This allows downstream consumers to provide their own LLVM.
    if any([m.is_root and m.name == "coralnpu-compiler" for m in module_ctx.modules]):
        new_local_repository(
            name = "llvm-raw",
            build_file_content = "# empty",
            path = "third_party/llvm-project",
        )

        # Keep in sync with ManageRV32Toolchain.cmake.
        http_archive(
            name = "rv32_toolchain",
            url = "https://github.com/riscv-collab/riscv-gnu-toolchain/releases/download/2026.08.27/riscv32-elf-ubuntu-22.04-llvm.tar.xz",
            sha256 = "e6f8a535563223ab2fe5ccc919145fa5f54d08d66eed18556c3179603b808f7e",
            strip_prefix = "riscv",
            build_file = "//build_tools/bazel:rv32_toolchain.BUILD",
        )

        # coralnpu_hw's repos with warnings silenced; keep in sync.
        # TODO: Silence warnings in coralnpu itself and drop these overrides.
        http_archive(
            name = "riscv_isa_sim",
            build_file = "@coralnpu_hw//third_party:spike.BUILD",
            patch_args = ["-p1"],
            patch_cmds = ["sed -i 's/$(CXXFLAGS)/$(CXXFLAGS) -w/; s/$(CFLAGS)/$(CFLAGS) -w/' Makefile.in"],
            patches = [
                "@coralnpu_hw//third_party/spike:0001-Add-mpause.patch",
                "@coralnpu_hw//third_party/spike:0002-Coral-Deviations.patch",
                "@coralnpu_hw//third_party/spike:0003-Dump-GPRs-on-EBREAK.patch",
                "@coralnpu_hw//third_party/spike:0004-Add-custom-CoralNPU-CSRs-and-update-MVENDORID-MARCHI.patch",
                "@coralnpu_hw//third_party/spike:0005-Force-logging-in-vcompress.patch",
                "@coralnpu_hw//third_party/spike:0006-Hardwire-misa-as-read-only-WARL.patch",
                "@coralnpu_hw//third_party/spike:0007-Rename-yield-macro-avoid-boost-std-conflict.patch",
                "@coralnpu_hw//third_party/spike:0008-Link-libstdc-explicitly-in-LIBS.patch",
                "@coralnpu_hw//third_party/spike:0009-Fix-Zvt-and-CoralNPU-mstatus-alignment.patch",
                Label("//:riscv-isa-sim-0001-zvt-mstatus-ms-initial.patch"),  # Local.
            ],
            sha256 = "064f4c1f22005899fa5106b822bfdc7034ffc9babc2f5821771589047f028a66",
            strip_prefix = "riscv-isa-sim-93a10ae685ac85bd3d8a62054f8f180a4f76fc82",
            urls = ["https://github.com/riscv-software-src/riscv-isa-sim/archive/93a10ae685ac85bd3d8a62054f8f180a4f76fc82.tar.gz"],
        )
        http_archive(
            name = "accellera_systemc",
            build_file = "@coralnpu_hw//third_party/systemc:systemc.BUILD",
            patch_cmds = ["sed -i '1i add_compile_options(-w)' CMakeLists.txt"],
            sha256 = "bfb309485a8ad35a08ee78827d1647a451ec5455767b25136e74522a6f41e0ea",
            strip_prefix = "systemc-2.3.4",
            urls = ["https://github.com/accellera-official/systemc/archive/refs/tags/2.3.4.tar.gz"],
        )
        http_archive(
            name = "verilator",
            build_file = "@coralnpu_hw//third_party/verilator:verilator.BUILD.bazel",
            patch_args = ["-p1"],
            patch_cmds = ["sed -i '1i #pragma clang system_header' include/verilated*.h"],
            patches = [
                "@coralnpu_hw//third_party/verilator:0001-Remove-autodetect-of-VERILATOR_ROOT.patch",
            ],
            sha256 = "8c8d2e11e6ad32f641dd250742a94195ddecb912e2e2dabe2f42ddbbb99c1092",
            strip_prefix = "verilator-5.052",
            urls = ["https://github.com/verilator/verilator/archive/refs/tags/v5.052.tar.gz"],
        )

coralnpuc_extension = module_extension(
    implementation = _coralnpuc_extension_impl,
)
