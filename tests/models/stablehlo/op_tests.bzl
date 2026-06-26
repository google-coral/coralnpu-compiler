"""Generated op_tests list for StableHLO."""

load("//tests/models/stablehlo:op_tests_f32.bzl", "stablehlo_op_tests_f32")
load("//tests/models/stablehlo:op_tests_i16.bzl", "stablehlo_op_tests_i16")
load("//tests/models/stablehlo:op_tests_i32.bzl", "stablehlo_op_tests_i32")
load("//tests/models/stablehlo:op_tests_i8.bzl", "stablehlo_op_tests_i8")

def stablehlo_op_tests(name = "stablehlo_op_tests"):
    """Registers all StableHLO op tests.

    Args:
      name: The name of the macro.
    """
    stablehlo_op_tests_i8()
    stablehlo_op_tests_i16()
    stablehlo_op_tests_i32()
    stablehlo_op_tests_f32()
