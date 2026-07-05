"""Generated op_tests list for Linalg."""

load("//tests/models/linalg:op_tests_f32.bzl", "linalg_op_tests_f32")
load("//tests/models/linalg:op_tests_i16.bzl", "linalg_op_tests_i16")
load("//tests/models/linalg:op_tests_i32.bzl", "linalg_op_tests_i32")
load("//tests/models/linalg:op_tests_i8.bzl", "linalg_op_tests_i8")

def linalg_op_tests(name = "linalg_op_tests"):
    """Registers all Linalg op tests.

    Args:
      name: The name of the macro.
    """
    linalg_op_tests_i8()
    linalg_op_tests_i16()
    linalg_op_tests_i32()
    linalg_op_tests_f32()
