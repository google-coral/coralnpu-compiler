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
        http_archive(
            name = "rv32_toolchain",
            url = "https://github.com/riscv-collab/riscv-gnu-toolchain/releases/download/2026.07.15/riscv32-elf-ubuntu-22.04-gcc.tar.xz",
            sha256 = "ae36abbec394b29643154c1b4a1322e829937d04e82f41b47f9c27d3bd68e543",
            strip_prefix = "riscv",
            build_file = "//build_tools/bazel:rv32_toolchain.BUILD",
        )

coralnpuc_extension = module_extension(
    implementation = _coralnpuc_extension_impl,
)
