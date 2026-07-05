"""Package-specific macros for CoralNPU Linalg compiler tests."""

load("//tests:defs.bzl", "coralnpu_check_gen_tests", "coralnpu_check_test")

_COMMON_COMPILER_FLAGS = [
    # configure the coralnpu target
    "--iree-hal-target-device=coralnpu",
    "--coralnpu-target-abi=ilp32",
    "--coralnpu-target-cpu-features=+m,+f,+zvl128b,+zve32f",
    "> /dev/null",
]

_COMMON_RUNNER_ARGS = [
    "--device=coralnpu",
]

def op_tests(
        name,
        test,
        instances,
        arg_gens = [],
        default_gen = None,
        timeout = "short",
        compiler_flags = None,
        runner_args = None,
        **kwargs):
    """Registers templated CoralNPU tests.

    Args:
      name: The name of the test target.
      test: The template MLIR file.
      instances: The shape instances.
      arg_gens: The JIT generator files.
      default_gen: Default generators.
      timeout: The test timeout.
      compiler_flags: Overrides for compiler flags.
      runner_args: Overrides for runner args.
      **kwargs: Extra arguments.
    """
    if compiler_flags == None:
        compiler_flags = _COMMON_COMPILER_FLAGS
    if runner_args == None:
        runner_args = _COMMON_RUNNER_ARGS
    coralnpu_check_gen_tests(
        name = name,
        test = test,
        arg_gens = arg_gens,
        default_gen = default_gen,
        instances = instances,
        compiler_flags = compiler_flags,
        runner_args = runner_args,
        timeout = timeout,
        **kwargs
    )

def op_check_test(name, src, **kwargs):
    """Registers a non-templated CoralNPU check test.

    Args:
      name: The name of the test target.
      src: The check MLIR file.
      **kwargs: Extra arguments.
    """
    coralnpu_check_test(
        name = name,
        src = src,
        compiler_flags = _COMMON_COMPILER_FLAGS,
        runner_args = _COMMON_RUNNER_ARGS,
        **kwargs
    )
