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
            url = "https://storage.googleapis.com/shodan-public-artifacts/toolchain_iree_rv32.tar.gz",
            sha256 = "01481183814cc66d6a8efb32681e2f74f5a7de321e93c81d563b65e64e3846a8",
            strip_prefix = "toolchain_iree_rv32imf",
            build_file = "//build_tools/bazel:rv32_toolchain.BUILD",
        )

coralnpuc_extension = module_extension(
    implementation = _coralnpuc_extension_impl,
)
