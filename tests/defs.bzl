"""Custom Bazel macros for CoralNPU compiler tests."""

load("@iree_core//build_tools/bazel:iree_bytecode_module.bzl", "iree_bytecode_module")
load("@iree_core//build_tools/bazel:native_binary.bzl", "native_test")
load("//tools/check_gen:def.bzl", "check_gen_tests")

def coralnpu_check_test(
        name,
        src,
        compiler_flags = [],
        runner_args = [],
        tags = [],
        timeout = None,
        deps = [],
        **kwargs):
    bytecode_module_name = name + "_bytecode_module"

    iree_bytecode_module(
        name = bytecode_module_name,
        src = src,
        compile_tool = "@iree_core//tools:iree-compile",
        flags = list(compiler_flags),
        tags = ["target=coralnpu"],
        deps = deps,
        visibility = ["//visibility:private"],
    )

    native_test(
        name = name,
        args = [
            "--module=$(location :%s)" % bytecode_module_name,
        ] + runner_args,
        data = [":%s" % bytecode_module_name],
        src = "@iree_core//tools:iree-check-module",  # Use absolute label to be safe
        tags = tags + ["driver=coralnpu", "target=coralnpu"],
        timeout = timeout,
        **kwargs
    )

STANDARD_DEFAULT_GEN = "//tools/check_gen/generators:sequential_vmfb"

def coralnpu_check_gen_tests(
        name,
        test,
        instances,
        arg_gens = [],
        default_gen = None,
        compiler_flags = [],
        runner_args = [],
        tags = [],
        timeout = None,
        deps = [],
        **kwargs):
    """Defines a test generator and test targets for templated tests.

    Args:
        name: Base name for the targets.
        test: The test function MLIR file.
        instances: List of instance shape strings.
        arg_gens: List of generator MLIR or VMFB files/targets.
        default_gen: Default generators.
        compiler_flags: Flags for the compiler.
        runner_args: Args for the runner.
        tags: Tags for the test targets.
        timeout: Timeout for the test targets.
        deps: Dependencies for the test targets.
        **kwargs: Passed to all targets.
    """
    if default_gen == None:
        default_gen = STANDARD_DEFAULT_GEN

    check_gen_tests(
        name = name,
        test = test,
        arg_gens = arg_gens,
        default_gen = default_gen,
        instances = instances,
        compiler_flags = compiler_flags,
        runner_args = runner_args,
        tags = tags + ["driver=coralnpu", "target=coralnpu"],
        bytecode_tags = ["target=coralnpu"],
        timeout = timeout,
        deps = deps,
        **kwargs
    )
