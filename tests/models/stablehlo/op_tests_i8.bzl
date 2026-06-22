"""Generated op_tests list for StableHLO i8 tests."""

load("//tests/models/stablehlo:defs.bzl", "op_tests")

def op_tests_i8(name, **kwargs):
    """Wrapper macro for op_tests with 'i8' and 'ci' tags.

    Args:
      name: The name of the test target.
      **kwargs: Additional arguments passed to op_tests.
    """
    tags = list(kwargs.pop("tags", []))
    if "i8" not in tags:
        tags.append("i8")
    if "ci" not in tags:
        tags.append("ci")
    op_tests(name = name, tags = tags, **kwargs)

def stablehlo_op_tests_i8(name = "stablehlo_op_i8_tests"):
    """Registers StableHLO i8 op tests.

    Args:
      name: The name of the macro.
    """
    op_tests_i8(name = "abs_rank1_i8", instances = ["(8)", "(256)", "(450)"], test = "abs_rank1_i8.mlir")
    op_tests_i8(name = "abs_rank2_i8", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "abs_rank2_i8.mlir")
    op_tests_i8(name = "abs_rank3_i8", instances = ["(2,3,4)", "(10,20,30)", "(5,100,2)"], test = "abs_rank3_i8.mlir")
    op_tests_i8(name = "abs_rank4_i8", instances = ["(2,2,3,2)", "(2,3,4,50)", "(1,1,5,400)"], test = "abs_rank4_i8.mlir")
    op_tests_i8(name = "add_rank1_i8", instances = ["(8)(8)", "(256)(256)", "(450)(450)"], test = "add_rank1_i8.mlir")
    op_tests_i8(name = "add_rank2_i8", instances = ["(4,8)(4,8)", "(120,256)(120,256)", "(300,450)(300,450)"], test = "add_rank2_i8.mlir")
    op_tests_i8(name = "add_rank3_i8", instances = ["(2,3,4)(2,3,4)", "(10,20,30)(10,20,30)", "(5,100,2)(5,100,2)"], test = "add_rank3_i8.mlir")
    op_tests_i8(name = "add_rank4_i8", instances = ["(2,2,3,2)(2,2,3,2)", "(2,3,4,50)(2,3,4,50)", "(1,1,5,400)(1,1,5,400)"], test = "add_rank4_i8.mlir")
    op_tests_i8(name = "broadcast_in_dim_rank2_i8_256", instances = ["(256)"], test = "broadcast_in_dim_rank2_i8_256.mlir")
    op_tests_i8(name = "broadcast_in_dim_rank2_i8_450", instances = ["(450)"], test = "broadcast_in_dim_rank2_i8_450.mlir")
    op_tests_i8(name = "broadcast_in_dim_rank2_i8_8", instances = ["(8)"], test = "broadcast_in_dim_rank2_i8_8.mlir")
    op_tests_i8(name = "broadcast_in_dim_rank3_i8_120_256", instances = ["(120,256)"], test = "broadcast_in_dim_rank3_i8_120_256.mlir")
    op_tests_i8(name = "broadcast_in_dim_rank3_i8_300_450", instances = ["(300,450)"], test = "broadcast_in_dim_rank3_i8_300_450.mlir")
    op_tests_i8(name = "broadcast_in_dim_rank3_i8_4_8", instances = ["(4,8)"], test = "broadcast_in_dim_rank3_i8_4_8.mlir")
    op_tests_i8(name = "broadcast_in_dim_rank4_i8_10_20_30", instances = ["(10,20,30)"], test = "broadcast_in_dim_rank4_i8_10_20_30.mlir")
    op_tests_i8(name = "broadcast_in_dim_rank4_i8_2_3_4", instances = ["(2,3,4)"], test = "broadcast_in_dim_rank4_i8_2_3_4.mlir")
    op_tests_i8(name = "broadcast_in_dim_rank4_i8_5_100_2", instances = ["(5,100,2)"], test = "broadcast_in_dim_rank4_i8_5_100_2.mlir")
    op_tests_i8(name = "compare_eq_rank1_i8", instances = ["(8)(8)", "(256)(256)", "(450)(450)"], test = "compare_eq_rank1_i8.mlir")
    op_tests_i8(name = "compare_eq_rank2_i8", instances = ["(4,8)(4,8)", "(120,256)(120,256)", "(300,450)(300,450)"], test = "compare_eq_rank2_i8.mlir")
    op_tests_i8(name = "compare_eq_rank3_i8", instances = ["(2,3,4)(2,3,4)", "(10,20,30)(10,20,30)", "(5,100,2)(5,100,2)"], test = "compare_eq_rank3_i8.mlir")
    op_tests_i8(name = "compare_eq_rank4_i8", instances = ["(2,2,3,2)(2,2,3,2)", "(2,3,4,50)(2,3,4,50)", "(1,1,5,400)(1,1,5,400)"], test = "compare_eq_rank4_i8.mlir")
    op_tests_i8(name = "concatenate_rank1_i8", instances = ["(8)(16)", "(256)(100)", "(450)(50)"], test = "concatenate_rank1_i8.mlir")
    op_tests_i8(name = "concatenate_rank2_i8", instances = ["(4,8)(4,16)", "(120,256)(120,100)", "(300,450)(300,50)"], test = "concatenate_rank2_i8.mlir")
    op_tests_i8(name = "concatenate_rank3_i8", instances = ["(2,3,4)(2,3,8)", "(10,20,30)(10,20,10)", "(5,100,2)(5,100,5)"], test = "concatenate_rank3_i8.mlir")
    op_tests_i8(name = "concatenate_rank4_i8", instances = ["(2,2,3,2)(2,2,3,4)", "(2,3,4,50)(2,3,4,10)", "(1,1,5,400)(1,1,5,100)"], test = "concatenate_rank4_i8.mlir")
    op_tests_i8(name = "convolution_rank2_i8", instances = ["(4,8)(8,4)", "(120,256)(256,300)", ("(300,100)(100,450)", ["manual"])], test = "convolution_rank2_i8.mlir")
    op_tests_i8(name = "convolution_rank3_i8", instances = ["(1,8,3)(3,3,16)", "(2,120,3)(5,3,32)", "(1,300,3)(3,3,8)"], test = "convolution_rank3_i8.mlir")
    op_tests_i8(name = "convolution_rank4_i8", instances = ["(1,8,8,3)(3,3,3,16)", "(2,12,12,3)(5,5,3,32)", "(1,50,50,3)(3,3,3,8)"], test = "convolution_rank4_i8.mlir")
    op_tests_i8(name = "divide_rank1_i8", instances = ["(8)(8)", "(256)(256)", "(450)(450)"], test = "divide_rank1_i8.mlir")
    op_tests_i8(name = "divide_rank2_i8", instances = ["(4,8)(4,8)", "(120,256)(120,256)", "(300,450)(300,450)"], test = "divide_rank2_i8.mlir")
    op_tests_i8(name = "divide_rank3_i8", instances = ["(2,3,4)(2,3,4)", "(10,20,30)(10,20,30)", "(5,100,2)(5,100,2)"], test = "divide_rank3_i8.mlir")
    op_tests_i8(name = "divide_rank4_i8", instances = ["(2,2,3,2)(2,2,3,2)", "(2,3,4,50)(2,3,4,50)", "(1,1,5,400)(1,1,5,400)"], test = "divide_rank4_i8.mlir")
    op_tests_i8(name = "dynamic_broadcast_in_dim_rank2_i8", instances = ["(8)", "(256)", "(450)"], test = "dynamic_broadcast_in_dim_rank2_i8.mlir")
    op_tests_i8(name = "dynamic_broadcast_in_dim_rank3_i8", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "dynamic_broadcast_in_dim_rank3_i8.mlir")
    op_tests_i8(name = "dynamic_broadcast_in_dim_rank4_i8", instances = ["(2,3,4)", "(10,20,30)", "(5,100,2)"], test = "dynamic_broadcast_in_dim_rank4_i8.mlir")
    op_tests_i8(name = "matmul_rank1_i8", instances = ["(8)(8)", "(256)(256)"], test = "matmul_rank1_i8.mlir")
    op_tests_i8(name = "matmul_rank2_i8", instances = ["(4,8)(8,4)", "(120,256)(256,300)", ("(300,100)(100,450)", ["manual"])], test = "matmul_rank2_i8.mlir")
    op_tests_i8(name = "maximum_rank1_i8", instances = ["(8)(8)", "(256)(256)", "(450)(450)"], test = "maximum_rank1_i8.mlir")
    op_tests_i8(name = "maximum_rank2_i8", instances = ["(4,8)(4,8)", "(120,256)(120,256)", "(300,450)(300,450)"], test = "maximum_rank2_i8.mlir")
    op_tests_i8(name = "maximum_rank3_i8", instances = ["(2,3,4)(2,3,4)", "(10,20,30)(10,20,30)", "(5,100,2)(5,100,2)"], test = "maximum_rank3_i8.mlir")
    op_tests_i8(name = "maximum_rank4_i8", instances = ["(2,2,3,2)(2,2,3,2)", "(2,3,4,50)(2,3,4,50)", "(1,1,5,400)(1,1,5,400)"], test = "maximum_rank4_i8.mlir")
    op_tests_i8(name = "minimum_rank1_i8", instances = ["(8)(8)", "(256)(256)", "(450)(450)"], test = "minimum_rank1_i8.mlir")
    op_tests_i8(name = "minimum_rank2_i8", instances = ["(4,8)(4,8)", "(120,256)(120,256)", "(300,450)(300,450)"], test = "minimum_rank2_i8.mlir")
    op_tests_i8(name = "minimum_rank3_i8", instances = ["(2,3,4)(2,3,4)", "(10,20,30)(10,20,30)", "(5,100,2)(5,100,2)"], test = "minimum_rank3_i8.mlir")
    op_tests_i8(name = "minimum_rank4_i8", instances = ["(2,2,3,2)(2,2,3,2)", "(2,3,4,50)(2,3,4,50)", "(1,1,5,400)(1,1,5,400)"], test = "minimum_rank4_i8.mlir")
    op_tests_i8(name = "multiply_rank1_i8", instances = ["(8)(8)", "(256)(256)", "(450)(450)"], test = "multiply_rank1_i8.mlir")
    op_tests_i8(name = "multiply_rank2_i8", instances = ["(4,8)(4,8)", "(120,256)(120,256)", "(300,450)(300,450)"], test = "multiply_rank2_i8.mlir")
    op_tests_i8(name = "multiply_rank3_i8", instances = ["(2,3,4)(2,3,4)", "(10,20,30)(10,20,30)", "(5,100,2)(5,100,2)"], test = "multiply_rank3_i8.mlir")
    op_tests_i8(name = "multiply_rank4_i8", instances = ["(2,2,3,2)(2,2,3,2)", "(2,3,4,50)(2,3,4,50)", "(1,1,5,400)(1,1,5,400)"], test = "multiply_rank4_i8.mlir")
    op_tests_i8(name = "negate_rank1_i8", instances = ["(8)", "(256)", "(450)"], test = "negate_rank1_i8.mlir")
    op_tests_i8(name = "negate_rank2_i8", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "negate_rank2_i8.mlir")
    op_tests_i8(name = "negate_rank3_i8", instances = ["(2,3,4)", "(10,20,30)", "(5,100,2)"], test = "negate_rank3_i8.mlir")
    op_tests_i8(name = "negate_rank4_i8", instances = ["(2,2,3,2)", "(2,3,4,50)", "(1,1,5,400)"], test = "negate_rank4_i8.mlir")
    op_tests_i8(name = "reduce_sum_rank1_i8", instances = ["(8)", "(256)", "(450)"], test = "reduce_sum_rank1_i8.mlir")
    op_tests_i8(name = "reduce_sum_rank2_i8", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "reduce_sum_rank2_i8.mlir")
    op_tests_i8(name = "reduce_sum_rank3_i8", instances = ["(2,3,4)", "(10,20,30)", "(5,100,2)"], test = "reduce_sum_rank3_i8.mlir")
    op_tests_i8(name = "reduce_sum_rank4_i8", instances = ["(2,2,3,2)", "(2,3,4,50)", "(1,1,5,400)"], test = "reduce_sum_rank4_i8.mlir")
    op_tests_i8(name = "select_rank1_i8", instances = ["(8)(8)(8)", "(256)(256)(256)", "(450)(450)(450)"], test = "select_rank1_i8.mlir")
    op_tests_i8(name = "select_rank2_i8", instances = ["(4,8)(4,8)(4,8)", "(120,256)(120,256)(120,256)", "(300,450)(300,450)(300,450)"], test = "select_rank2_i8.mlir")
    op_tests_i8(name = "select_rank3_i8", instances = ["(2,3,4)(2,3,4)(2,3,4)", "(10,20,30)(10,20,30)(10,20,30)", "(5,100,2)(5,100,2)(5,100,2)"], test = "select_rank3_i8.mlir")
    op_tests_i8(name = "select_rank4_i8", instances = ["(2,2,3,2)(2,2,3,2)(2,2,3,2)", "(2,3,4,50)(2,3,4,50)(2,3,4,50)", "(1,1,5,400)(1,1,5,400)(1,1,5,400)"], test = "select_rank4_i8.mlir")
    op_tests_i8(name = "slice_rank1_i8", instances = ["(8)", "(256)", "(450)"], test = "slice_rank1_i8.mlir")
    op_tests_i8(name = "slice_rank2_i8", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "slice_rank2_i8.mlir")
    op_tests_i8(name = "slice_rank3_i8", instances = ["(2,3,4)", "(10,20,30)", "(5,100,4)"], test = "slice_rank3_i8.mlir")
    op_tests_i8(name = "slice_rank4_i8", instances = ["(2,2,3,4)", "(2,3,4,50)", "(2,2,5,400)"], test = "slice_rank4_i8.mlir")
    op_tests_i8(name = "subtract_rank1_i8", instances = ["(8)(8)", "(256)(256)", "(450)(450)"], test = "subtract_rank1_i8.mlir")
    op_tests_i8(name = "subtract_rank2_i8", instances = ["(4,8)(4,8)", "(120,256)(120,256)", "(300,450)(300,450)"], test = "subtract_rank2_i8.mlir")
    op_tests_i8(name = "subtract_rank3_i8", instances = ["(2,3,4)(2,3,4)", "(10,20,30)(10,20,30)", "(5,100,2)(5,100,2)"], test = "subtract_rank3_i8.mlir")
    op_tests_i8(name = "subtract_rank4_i8", instances = ["(2,2,3,2)(2,2,3,2)", "(2,3,4,50)(2,3,4,50)", "(1,1,5,400)(1,1,5,400)"], test = "subtract_rank4_i8.mlir")
    op_tests_i8(name = "transpose_rank2_i8", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "transpose_rank2_i8.mlir")
    op_tests_i8(name = "transpose_rank3_i8", instances = ["(2,3,4)", "(10,20,30)", "(5,100,2)"], test = "transpose_rank3_i8.mlir")
    op_tests_i8(name = "transpose_rank4_i8", instances = ["(2,2,3,2)", "(2,3,4,50)", "(1,1,5,400)"], test = "transpose_rank4_i8.mlir")

    # Elementwise operations

    # Reason: VMVX reference ctlz promotion bug (returns 32 instead of 8 for 0)
    op_tests_i8(name = "count_leading_zeros_rank1_i8", instances = ["(8)", "(256)", "(450)"], test = "count_leading_zeros_rank1_i8.mlir", tags = ["manual"])

    # Reason: VMVX reference ctlz promotion bug (returns 32 instead of 8 for 0)
    op_tests_i8(name = "count_leading_zeros_rank2_i8", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "count_leading_zeros_rank2_i8.mlir", tags = ["manual"])

    # Reason: VMVX reference ctlz promotion bug (returns 32 instead of 8 for 0)
    op_tests_i8(name = "count_leading_zeros_rank3_i8", instances = ["(2,3,4)", "(10,20,30)", "(5,100,2)"], test = "count_leading_zeros_rank3_i8.mlir", tags = ["manual"])

    # Reason: VMVX reference ctlz promotion bug (returns 32 instead of 8 for 0)
    op_tests_i8(name = "count_leading_zeros_rank4_i8", instances = ["(2,2,3,2)", "(2,3,4,50)", "(1,1,5,400)"], test = "count_leading_zeros_rank4_i8.mlir", tags = ["manual"])
    op_tests_i8(name = "not_rank1_i8", instances = ["(8)", "(256)", "(450)"], test = "not_rank1_i8.mlir")
    op_tests_i8(name = "not_rank2_i8", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "not_rank2_i8.mlir")
    op_tests_i8(name = "not_rank3_i8", instances = ["(2,3,4)", "(10,20,30)", "(5,100,2)"], test = "not_rank3_i8.mlir")
    op_tests_i8(name = "not_rank4_i8", instances = ["(2,2,3,2)", "(2,3,4,50)", "(1,1,5,400)"], test = "not_rank4_i8.mlir")

    # Reason: VMVX lacks math.ctpop legalization
    op_tests_i8(name = "popcnt_rank1_i8", instances = ["(8)", "(256)", "(450)"], test = "popcnt_rank1_i8.mlir", tags = ["manual"])

    # Reason: VMVX lacks math.ctpop legalization
    op_tests_i8(name = "popcnt_rank2_i8", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "popcnt_rank2_i8.mlir", tags = ["manual"])

    # Reason: VMVX lacks math.ctpop legalization
    op_tests_i8(name = "popcnt_rank3_i8", instances = ["(2,3,4)", "(10,20,30)", "(5,100,2)"], test = "popcnt_rank3_i8.mlir", tags = ["manual"])

    # Reason: VMVX lacks math.ctpop legalization
    op_tests_i8(name = "popcnt_rank4_i8", instances = ["(2,2,3,2)", "(2,3,4,50)", "(1,1,5,400)"], test = "popcnt_rank4_i8.mlir", tags = ["manual"])
    op_tests_i8(name = "sign_rank1_i8", instances = ["(8)", "(256)", "(450)"], test = "sign_rank1_i8.mlir")
    op_tests_i8(name = "sign_rank2_i8", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "sign_rank2_i8.mlir")
    op_tests_i8(name = "sign_rank3_i8", instances = ["(2,3,4)", "(10,20,30)", "(5,100,2)"], test = "sign_rank3_i8.mlir")
    op_tests_i8(name = "sign_rank4_i8", instances = ["(2,2,3,2)", "(2,3,4,50)", "(1,1,5,400)"], test = "sign_rank4_i8.mlir")
    op_tests_i8(name = "and_rank1_i8", instances = ["(8)(8)", "(256)(256)", "(450)(450)"], test = "and_rank1_i8.mlir")
    op_tests_i8(name = "and_rank2_i8", instances = ["(4,8)(4,8)", "(120,256)(120,256)", "(300,450)(300,450)"], test = "and_rank2_i8.mlir")
    op_tests_i8(name = "and_rank3_i8", instances = ["(2,3,4)(2,3,4)", "(10,20,30)(10,20,30)", "(5,100,2)(5,100,2)"], test = "and_rank3_i8.mlir")
    op_tests_i8(name = "and_rank4_i8", instances = ["(2,2,3,2)(2,2,3,2)", "(2,3,4,50)(2,3,4,50)", "(1,1,5,400)(1,1,5,400)"], test = "and_rank4_i8.mlir")
    op_tests_i8(name = "or_rank1_i8", instances = ["(8)(8)", "(256)(256)", "(450)(450)"], test = "or_rank1_i8.mlir")
    op_tests_i8(name = "or_rank2_i8", instances = ["(4,8)(4,8)", "(120,256)(120,256)", "(300,450)(300,450)"], test = "or_rank2_i8.mlir")
    op_tests_i8(name = "or_rank3_i8", instances = ["(2,3,4)(2,3,4)", "(10,20,30)(10,20,30)", "(5,100,2)(5,100,2)"], test = "or_rank3_i8.mlir")
    op_tests_i8(name = "or_rank4_i8", instances = ["(2,2,3,2)(2,2,3,2)", "(2,3,4,50)(2,3,4,50)", "(1,1,5,400)(1,1,5,400)"], test = "or_rank4_i8.mlir")
    op_tests_i8(name = "xor_rank1_i8", instances = ["(8)(8)", "(256)(256)", "(450)(450)"], test = "xor_rank1_i8.mlir")
    op_tests_i8(name = "xor_rank2_i8", instances = ["(4,8)(4,8)", "(120,256)(120,256)", "(300,450)(300,450)"], test = "xor_rank2_i8.mlir")
    op_tests_i8(name = "xor_rank3_i8", instances = ["(2,3,4)(2,3,4)", "(10,20,30)(10,20,30)", "(5,100,2)(5,100,2)"], test = "xor_rank3_i8.mlir")
    op_tests_i8(name = "xor_rank4_i8", instances = ["(2,2,3,2)(2,2,3,2)", "(2,3,4,50)(2,3,4,50)", "(1,1,5,400)(1,1,5,400)"], test = "xor_rank4_i8.mlir")
    op_tests_i8(name = "shift_left_rank1_i8", instances = ["(8)(8)", "(256)(256)", "(450)(450)"], test = "shift_left_rank1_i8.mlir")
    op_tests_i8(name = "shift_left_rank2_i8", instances = ["(4,8)(4,8)", "(120,256)(120,256)", "(300,450)(300,450)"], test = "shift_left_rank2_i8.mlir")
    op_tests_i8(name = "shift_left_rank3_i8", instances = ["(2,3,4)(2,3,4)", "(10,20,30)(10,20,30)", "(5,100,2)(5,100,2)"], test = "shift_left_rank3_i8.mlir")
    op_tests_i8(name = "shift_left_rank4_i8", instances = ["(2,2,3,2)(2,2,3,2)", "(2,3,4,50)(2,3,4,50)", "(1,1,5,400)(1,1,5,400)"], test = "shift_left_rank4_i8.mlir")
    op_tests_i8(name = "shift_right_arithmetic_rank1_i8", instances = ["(8)(8)", "(256)(256)", "(450)(450)"], test = "shift_right_arithmetic_rank1_i8.mlir")
    op_tests_i8(name = "shift_right_arithmetic_rank2_i8", instances = ["(4,8)(4,8)", "(120,256)(120,256)", "(300,450)(300,450)"], test = "shift_right_arithmetic_rank2_i8.mlir")
    op_tests_i8(name = "shift_right_arithmetic_rank3_i8", instances = ["(2,3,4)(2,3,4)", "(10,20,30)(10,20,30)", "(5,100,2)(5,100,2)"], test = "shift_right_arithmetic_rank3_i8.mlir")
    op_tests_i8(name = "shift_right_arithmetic_rank4_i8", instances = ["(2,2,3,2)(2,2,3,2)", "(2,3,4,50)(2,3,4,50)", "(1,1,5,400)(1,1,5,400)"], test = "shift_right_arithmetic_rank4_i8.mlir")
    op_tests_i8(name = "shift_right_logical_rank1_i8", instances = ["(8)(8)", "(256)(256)", "(450)(450)"], test = "shift_right_logical_rank1_i8.mlir")
    op_tests_i8(name = "shift_right_logical_rank2_i8", instances = ["(4,8)(4,8)", "(120,256)(120,256)", "(300,450)(300,450)"], test = "shift_right_logical_rank2_i8.mlir")
    op_tests_i8(name = "shift_right_logical_rank3_i8", instances = ["(2,3,4)(2,3,4)", "(10,20,30)(10,20,30)", "(5,100,2)(5,100,2)"], test = "shift_right_logical_rank3_i8.mlir")
    op_tests_i8(name = "shift_right_logical_rank4_i8", instances = ["(2,2,3,2)(2,2,3,2)", "(2,3,4,50)(2,3,4,50)", "(1,1,5,400)(1,1,5,400)"], test = "shift_right_logical_rank4_i8.mlir")
    op_tests_i8(name = "power_rank1_i8", instances = ["(8)(8)", "(256)(256)", "(450)(450)"], test = "power_rank1_i8.mlir")
    op_tests_i8(name = "power_rank2_i8", instances = ["(4,8)(4,8)", "(120,256)(120,256)", "(300,450)(300,450)"], test = "power_rank2_i8.mlir")
    op_tests_i8(name = "power_rank3_i8", instances = ["(2,3,4)(2,3,4)", "(10,20,30)(10,20,30)", "(5,100,2)(5,100,2)"], test = "power_rank3_i8.mlir")
    op_tests_i8(name = "power_rank4_i8", instances = ["(2,2,3,2)(2,2,3,2)", "(2,3,4,50)(2,3,4,50)", "(1,1,5,400)(1,1,5,400)"], test = "power_rank4_i8.mlir")
    op_tests_i8(name = "remainder_rank1_i8", instances = ["(8)(8)", "(256)(256)", "(450)(450)"], test = "remainder_rank1_i8.mlir")
    op_tests_i8(name = "remainder_rank2_i8", instances = ["(4,8)(4,8)", "(120,256)(120,256)", "(300,450)(300,450)"], test = "remainder_rank2_i8.mlir")
    op_tests_i8(name = "remainder_rank3_i8", instances = ["(2,3,4)(2,3,4)", "(10,20,30)(10,20,30)", "(5,100,2)(5,100,2)"], test = "remainder_rank3_i8.mlir")
    op_tests_i8(name = "remainder_rank4_i8", instances = ["(2,2,3,2)(2,2,3,2)", "(2,3,4,50)(2,3,4,50)", "(1,1,5,400)(1,1,5,400)"], test = "remainder_rank4_i8.mlir")
