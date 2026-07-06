"""Generated op_tests list for StableHLO f32 tests."""

load("//tests/models/stablehlo:defs.bzl", "op_tests")

def op_tests_f32(name, **kwargs):
    """Wrapper macro for op_tests with 'f32' and 'ci' tags.

    Args:
      name: The name of the test target.
      **kwargs: Additional arguments passed to op_tests.
    """
    tags = list(kwargs.pop("tags", []))
    if "f32" not in tags:
        tags.append("f32")
    if "ci" not in tags:
        tags.append("ci")
    op_tests(name = name, tags = tags, **kwargs)

def stablehlo_op_tests_f32(name = "stablehlo_op_f32_tests"):
    """Registers StableHLO f32 op tests.

    Args:
      name: The name of the macro.
    """
    op_tests_f32(name = "abs_rank1_f32", instances = ["(8)", "(256)", "(450)"], test = "abs_rank1_f32.mlir")
    op_tests_f32(name = "abs_rank2_f32", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "abs_rank2_f32.mlir")
    op_tests_f32(name = "abs_rank3_f32", instances = ["(2,3,4)", "(10,20,30)", "(5,100,2)"], test = "abs_rank3_f32.mlir")
    op_tests_f32(name = "abs_rank4_f32", instances = ["(2,2,3,2)", "(2,3,4,50)", "(1,1,5,400)"], test = "abs_rank4_f32.mlir")
    op_tests_f32(name = "add_rank1_f32", instances = ["(8)(8)", "(256)(256)", "(450)(450)"], test = "add_rank1_f32.mlir")
    op_tests_f32(name = "add_rank2_f32", instances = ["(4,8)(4,8)", "(120,256)(120,256)", "(300,450)(300,450)"], test = "add_rank2_f32.mlir")
    op_tests_f32(name = "add_rank3_f32", instances = ["(2,3,4)(2,3,4)", "(10,20,30)(10,20,30)", "(5,100,2)(5,100,2)"], test = "add_rank3_f32.mlir")
    op_tests_f32(name = "add_rank4_f32", instances = ["(2,2,3,2)(2,2,3,2)", "(2,3,4,50)(2,3,4,50)", "(1,1,5,400)(1,1,5,400)"], test = "add_rank4_f32.mlir")
    op_tests_f32(name = "broadcast_in_dim_rank2_f32_256", instances = ["(256)"], test = "broadcast_in_dim_rank2_f32_256.mlir")
    op_tests_f32(name = "broadcast_in_dim_rank2_f32_450", instances = ["(450)"], test = "broadcast_in_dim_rank2_f32_450.mlir")
    op_tests_f32(name = "broadcast_in_dim_rank2_f32_8", instances = ["(8)"], test = "broadcast_in_dim_rank2_f32_8.mlir")
    op_tests_f32(name = "broadcast_in_dim_rank3_f32_120_256", instances = ["(120,256)"], test = "broadcast_in_dim_rank3_f32_120_256.mlir")
    op_tests_f32(name = "broadcast_in_dim_rank3_f32_300_450", instances = ["(300,450)"], test = "broadcast_in_dim_rank3_f32_300_450.mlir")
    op_tests_f32(name = "broadcast_in_dim_rank3_f32_4_8", instances = ["(4,8)"], test = "broadcast_in_dim_rank3_f32_4_8.mlir")
    op_tests_f32(name = "broadcast_in_dim_rank4_f32_10_20_30", instances = ["(10,20,30)"], test = "broadcast_in_dim_rank4_f32_10_20_30.mlir")
    op_tests_f32(name = "broadcast_in_dim_rank4_f32_2_3_4", instances = ["(2,3,4)"], test = "broadcast_in_dim_rank4_f32_2_3_4.mlir")
    op_tests_f32(name = "broadcast_in_dim_rank4_f32_5_100_2", instances = ["(5,100,2)"], test = "broadcast_in_dim_rank4_f32_5_100_2.mlir")
    op_tests_f32(name = "compare_eq_rank1_f32", instances = ["(8)(8)", "(256)(256)", "(450)(450)"], test = "compare_eq_rank1_f32.mlir")
    op_tests_f32(name = "compare_eq_rank2_f32", instances = ["(4,8)(4,8)", "(120,256)(120,256)", "(300,450)(300,450)"], test = "compare_eq_rank2_f32.mlir")
    op_tests_f32(name = "compare_eq_rank3_f32", instances = ["(2,3,4)(2,3,4)", "(10,20,30)(10,20,30)", "(5,100,2)(5,100,2)"], test = "compare_eq_rank3_f32.mlir")
    op_tests_f32(name = "compare_eq_rank4_f32", instances = ["(2,2,3,2)(2,2,3,2)", "(2,3,4,50)(2,3,4,50)", "(1,1,5,400)(1,1,5,400)"], test = "compare_eq_rank4_f32.mlir")
    op_tests_f32(name = "concatenate_rank1_f32", instances = ["(8)(16)", "(256)(100)", "(450)(50)"], test = "concatenate_rank1_f32.mlir")
    op_tests_f32(name = "concatenate_rank2_f32", instances = ["(4,8)(4,16)", "(120,256)(120,100)", "(300,450)(300,50)"], test = "concatenate_rank2_f32.mlir")
    op_tests_f32(name = "concatenate_rank3_f32", instances = ["(2,3,4)(2,3,8)", "(10,20,30)(10,20,10)", "(5,100,2)(5,100,5)"], test = "concatenate_rank3_f32.mlir")
    op_tests_f32(name = "concatenate_rank4_f32", instances = ["(2,2,3,2)(2,2,3,4)", "(2,3,4,50)(2,3,4,10)", "(1,1,5,400)(1,1,5,100)"], test = "concatenate_rank4_f32.mlir")
    op_tests_f32(name = "convolution_rank2_f32", instances = ["(4,8)(8,4)", "(120,256)(256,300)", "(300,100)(100,450)"], test = "convolution_rank2_f32.mlir")
    op_tests_f32(name = "convolution_rank3_f32", instances = ["(1,8,3)(3,3,16)", "(2,120,3)(5,3,32)", "(1,300,3)(3,3,8)"], test = "convolution_rank3_f32.mlir")
    op_tests_f32(name = "convolution_rank4_f32", instances = ["(1,8,8,3)(3,3,3,16)"], test = "convolution_rank4_f32.mlir")
    op_tests_f32(name = "divide_rank1_f32", instances = ["(8)(8)", "(256)(256)", "(450)(450)"], test = "divide_rank1_f32.mlir")
    op_tests_f32(name = "divide_rank2_f32", instances = ["(4,8)(4,8)", "(120,256)(120,256)", "(300,450)(300,450)"], test = "divide_rank2_f32.mlir")
    op_tests_f32(name = "divide_rank3_f32", instances = ["(2,3,4)(2,3,4)", "(10,20,30)(10,20,30)", "(5,100,2)(5,100,2)"], test = "divide_rank3_f32.mlir")
    op_tests_f32(name = "divide_rank4_f32", instances = ["(2,2,3,2)(2,2,3,2)", "(2,3,4,50)(2,3,4,50)", "(1,1,5,400)(1,1,5,400)"], test = "divide_rank4_f32.mlir")
    op_tests_f32(name = "dynamic_broadcast_in_dim_rank2_f32", instances = ["(8)", "(256)", "(450)"], test = "dynamic_broadcast_in_dim_rank2_f32.mlir")
    op_tests_f32(name = "dynamic_broadcast_in_dim_rank3_f32", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "dynamic_broadcast_in_dim_rank3_f32.mlir")
    op_tests_f32(name = "dynamic_broadcast_in_dim_rank4_f32", instances = ["(2,3,4)", "(10,20,30)", "(5,100,2)"], test = "dynamic_broadcast_in_dim_rank4_f32.mlir")
    op_tests_f32(name = "matmul_rank1_f32", instances = ["(8)(8)", "(256)(256)"], test = "matmul_rank1_f32.mlir")
    op_tests_f32(name = "matmul_rank2_f32", instances = ["(4,8)(8,4)", "(120,256)(256,300)", "(300,100)(100,450)"], test = "matmul_rank2_f32.mlir")
    op_tests_f32(name = "maximum_rank1_f32", instances = ["(8)(8)", "(256)(256)", "(450)(450)"], test = "maximum_rank1_f32.mlir")
    op_tests_f32(name = "maximum_rank2_f32", instances = ["(4,8)(4,8)", "(120,256)(120,256)", "(300,450)(300,450)"], test = "maximum_rank2_f32.mlir")
    op_tests_f32(name = "maximum_rank3_f32", instances = ["(2,3,4)(2,3,4)", "(10,20,30)(10,20,30)", "(5,100,2)(5,100,2)"], test = "maximum_rank3_f32.mlir")
    op_tests_f32(name = "maximum_rank4_f32", instances = ["(2,2,3,2)(2,2,3,2)", "(2,3,4,50)(2,3,4,50)", "(1,1,5,400)(1,1,5,400)"], test = "maximum_rank4_f32.mlir")
    op_tests_f32(name = "minimum_rank1_f32", instances = ["(8)(8)", "(256)(256)", "(450)(450)"], test = "minimum_rank1_f32.mlir")
    op_tests_f32(name = "minimum_rank2_f32", instances = ["(4,8)(4,8)", "(120,256)(120,256)", "(300,450)(300,450)"], test = "minimum_rank2_f32.mlir")
    op_tests_f32(name = "minimum_rank3_f32", instances = ["(2,3,4)(2,3,4)", "(10,20,30)(10,20,30)", "(5,100,2)(5,100,2)"], test = "minimum_rank3_f32.mlir")
    op_tests_f32(name = "minimum_rank4_f32", instances = ["(2,2,3,2)(2,2,3,2)", "(2,3,4,50)(2,3,4,50)", "(1,1,5,400)(1,1,5,400)"], test = "minimum_rank4_f32.mlir")
    op_tests_f32(name = "multiply_rank1_f32", instances = ["(8)(8)", "(256)(256)", "(450)(450)"], test = "multiply_rank1_f32.mlir")
    op_tests_f32(name = "multiply_rank2_f32", instances = ["(4,8)(4,8)", "(120,256)(120,256)", "(300,450)(300,450)"], test = "multiply_rank2_f32.mlir")
    op_tests_f32(name = "multiply_rank3_f32", instances = ["(2,3,4)(2,3,4)", "(10,20,30)(10,20,30)", "(5,100,2)(5,100,2)"], test = "multiply_rank3_f32.mlir")
    op_tests_f32(name = "multiply_rank4_f32", instances = ["(2,2,3,2)(2,2,3,2)", "(2,3,4,50)(2,3,4,50)", "(1,1,5,400)(1,1,5,400)"], test = "multiply_rank4_f32.mlir")
    op_tests_f32(name = "negate_rank1_f32", instances = ["(8)", "(256)", "(450)"], test = "negate_rank1_f32.mlir")
    op_tests_f32(name = "negate_rank2_f32", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "negate_rank2_f32.mlir")
    op_tests_f32(name = "negate_rank3_f32", instances = ["(2,3,4)", "(10,20,30)", "(5,100,2)"], test = "negate_rank3_f32.mlir")
    op_tests_f32(name = "negate_rank4_f32", instances = ["(2,2,3,2)", "(2,3,4,50)", "(1,1,5,400)"], test = "negate_rank4_f32.mlir")
    op_tests_f32(name = "reduce_sum_rank1_f32", instances = ["(8)", "(256)", "(450)"], test = "reduce_sum_rank1_f32.mlir")
    op_tests_f32(name = "reduce_sum_rank2_f32", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "reduce_sum_rank2_f32.mlir")
    op_tests_f32(name = "reduce_sum_rank3_f32", instances = ["(2,3,4)", "(10,20,30)", "(5,100,2)"], test = "reduce_sum_rank3_f32.mlir")
    op_tests_f32(name = "reduce_sum_rank4_f32", instances = ["(2,2,3,2)", "(2,3,4,50)", "(1,1,5,400)"], test = "reduce_sum_rank4_f32.mlir")
    op_tests_f32(name = "select_rank1_f32", instances = ["(8)(8)(8)", "(256)(256)(256)", "(450)(450)(450)"], test = "select_rank1_f32.mlir")
    op_tests_f32(name = "select_rank2_f32", instances = ["(4,8)(4,8)(4,8)", "(120,256)(120,256)(120,256)", "(300,450)(300,450)(300,450)"], test = "select_rank2_f32.mlir")
    op_tests_f32(name = "select_rank3_f32", instances = ["(2,3,4)(2,3,4)(2,3,4)", "(10,20,30)(10,20,30)(10,20,30)", "(5,100,2)(5,100,2)(5,100,2)"], test = "select_rank3_f32.mlir")
    op_tests_f32(name = "select_rank4_f32", instances = ["(2,2,3,2)(2,2,3,2)(2,2,3,2)", "(2,3,4,50)(2,3,4,50)(2,3,4,50)", "(1,1,5,400)(1,1,5,400)(1,1,5,400)"], test = "select_rank4_f32.mlir")
    op_tests_f32(name = "slice_rank1_f32", instances = ["(8)", "(256)", "(450)"], test = "slice_rank1_f32.mlir")
    op_tests_f32(name = "slice_rank2_f32", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "slice_rank2_f32.mlir")
    op_tests_f32(name = "slice_rank3_f32", instances = ["(2,3,4)", "(10,20,30)", "(5,100,4)"], test = "slice_rank3_f32.mlir")
    op_tests_f32(name = "slice_rank4_f32", instances = ["(2,2,3,4)", "(2,3,4,50)", "(2,2,5,400)"], test = "slice_rank4_f32.mlir")
    op_tests_f32(name = "subtract_rank1_f32", instances = ["(8)(8)", "(256)(256)", "(450)(450)"], test = "subtract_rank1_f32.mlir")
    op_tests_f32(name = "subtract_rank2_f32", instances = ["(4,8)(4,8)", "(120,256)(120,256)", "(300,450)(300,450)"], test = "subtract_rank2_f32.mlir")
    op_tests_f32(name = "subtract_rank3_f32", instances = ["(2,3,4)(2,3,4)", "(10,20,30)(10,20,30)", "(5,100,2)(5,100,2)"], test = "subtract_rank3_f32.mlir")
    op_tests_f32(name = "subtract_rank4_f32", instances = ["(2,2,3,2)(2,2,3,2)", "(2,3,4,50)(2,3,4,50)", "(1,1,5,400)(1,1,5,400)"], test = "subtract_rank4_f32.mlir")
    op_tests_f32(name = "transpose_rank2_f32", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "transpose_rank2_f32.mlir")
    op_tests_f32(name = "transpose_rank3_f32", instances = ["(2,3,4)", "(10,20,30)", "(5,100,2)"], test = "transpose_rank3_f32.mlir")
    op_tests_f32(name = "transpose_rank4_f32", instances = ["(2,2,3,2)", "(2,3,4,50)", "(1,1,5,400)"], test = "transpose_rank4_f32.mlir")
    op_tests_f32(name = "pad_rank2_f32", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "pad_rank2_f32.mlir")
    op_tests_f32(name = "clamp_rank2_f32", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "clamp_rank2_f32.mlir")
    op_tests_f32(name = "broadcast_rank2_f32", instances = ["(8)", "(256)", "(450)"], test = "broadcast_rank2_f32.mlir")
    op_tests_f32(name = "if_rank1_f32", instances = ["(1)(8)", "(1)(256)"], test = "if_rank1_f32.mlir")
    op_tests_f32(name = "case_rank1_f32", instances = ["(1)(8)", "(1)(256)"], test = "case_rank1_f32.mlir")
    op_tests_f32(name = "dot_general_rank3_f32", instances = ["(2,2,3)(2,3,4)", "(1,8,16)(1,16,8)"], test = "dot_general_rank3_f32.mlir")
    op_tests_f32(name = "gather_rank1_f32", instances = ["(10)(8)", "(300)(256)"], test = "gather_rank1_f32.mlir")
    op_tests_f32(name = "scatter_rank1_f32", instances = ["(10)(3)(3)", "(300)(256)(256)"], test = "scatter_rank1_f32.mlir")
    op_tests_f32(name = "sort_rank1_f32", instances = ["(8)", "(256)"], test = "sort_rank1_f32.mlir")
    op_tests_f32(name = "tuple_rank1_f32", instances = ["(8)(8)", "(256)(256)"], test = "tuple_rank1_f32.mlir")
    op_tests_f32(name = "bitcast_convert_rank1_f32_i32", instances = ["(8)", "(256)"], test = "bitcast_convert_rank1_f32_i32.mlir")
    op_tests_f32(name = "batch_norm_inference_rank4_f32", instances = ["(1,2,2,3)(3)(3)(3)(3)", "(1,8,16,32)(32)(32)(32)(32)"], test = "batch_norm_inference_rank4_f32.mlir")

    # Elementwise operations

    # Reason: VMVX lacks math.cbrt legalization
    op_tests_f32(name = "cbrt_rank1_f32", instances = ["(8)", "(256)", "(450)"], test = "cbrt_rank1_f32.mlir", tags = ["manual"])

    # Reason: VMVX lacks math.cbrt legalization
    op_tests_f32(name = "cbrt_rank2_f32", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "cbrt_rank2_f32.mlir", tags = ["manual"])

    # Reason: VMVX lacks math.cbrt legalization
    op_tests_f32(name = "cbrt_rank3_f32", instances = ["(2,3,4)", "(10,20,30)", "(5,100,2)"], test = "cbrt_rank3_f32.mlir", tags = ["manual"])

    # Reason: VMVX lacks math.cbrt legalization
    op_tests_f32(name = "cbrt_rank4_f32", instances = ["(2,2,3,2)", "(2,3,4,50)", "(1,1,5,400)"], test = "cbrt_rank4_f32.mlir", tags = ["manual"])
    op_tests_f32(name = "ceil_rank1_f32", instances = ["(8)", "(256)", "(450)"], test = "ceil_rank1_f32.mlir")
    op_tests_f32(name = "ceil_rank2_f32", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "ceil_rank2_f32.mlir")
    op_tests_f32(name = "ceil_rank3_f32", instances = ["(2,3,4)", "(10,20,30)", "(5,100,2)"], test = "ceil_rank3_f32.mlir")
    op_tests_f32(name = "ceil_rank4_f32", instances = ["(2,2,3,2)", "(2,3,4,50)", "(1,1,5,400)"], test = "ceil_rank4_f32.mlir")
    op_tests_f32(name = "cosine_rank1_f32", instances = ["(8)", "(256)", "(450)"], test = "cosine_rank1_f32.mlir")
    op_tests_f32(name = "cosine_rank2_f32", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "cosine_rank2_f32.mlir")
    op_tests_f32(name = "cosine_rank3_f32", instances = ["(2,3,4)", "(10,20,30)", "(5,100,2)"], test = "cosine_rank3_f32.mlir")
    op_tests_f32(name = "cosine_rank4_f32", instances = ["(2,2,3,2)", "(2,3,4,50)", "(1,1,5,400)"], test = "cosine_rank4_f32.mlir")
    op_tests_f32(name = "exponential_rank1_f32", instances = ["(8)", "(256)", "(450)"], test = "exponential_rank1_f32.mlir")
    op_tests_f32(name = "exponential_rank2_f32", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "exponential_rank2_f32.mlir")
    op_tests_f32(name = "exponential_rank3_f32", instances = ["(2,3,4)", "(10,20,30)", "(5,100,2)"], test = "exponential_rank3_f32.mlir")
    op_tests_f32(name = "exponential_rank4_f32", instances = ["(2,2,3,2)", "(2,3,4,50)", "(1,1,5,400)"], test = "exponential_rank4_f32.mlir")
    op_tests_f32(name = "exponential_minus_one_rank1_f32", instances = ["(8)", "(256)", "(450)"], test = "exponential_minus_one_rank1_f32.mlir")
    op_tests_f32(name = "exponential_minus_one_rank2_f32", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "exponential_minus_one_rank2_f32.mlir")
    op_tests_f32(name = "exponential_minus_one_rank3_f32", instances = ["(2,3,4)", "(10,20,30)", "(5,100,2)"], test = "exponential_minus_one_rank3_f32.mlir")
    op_tests_f32(name = "exponential_minus_one_rank4_f32", instances = ["(2,2,3,2)", "(2,3,4,50)", "(1,1,5,400)"], test = "exponential_minus_one_rank4_f32.mlir")
    op_tests_f32(name = "floor_rank1_f32", instances = ["(8)", "(256)", "(450)"], test = "floor_rank1_f32.mlir")
    op_tests_f32(name = "floor_rank2_f32", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "floor_rank2_f32.mlir")
    op_tests_f32(name = "floor_rank3_f32", instances = ["(2,3,4)", "(10,20,30)", "(5,100,2)"], test = "floor_rank3_f32.mlir")
    op_tests_f32(name = "floor_rank4_f32", instances = ["(2,2,3,2)", "(2,3,4,50)", "(1,1,5,400)"], test = "floor_rank4_f32.mlir")
    op_tests_f32(name = "log_rank1_f32", instances = ["(8)", "(256)", "(450)"], test = "log_rank1_f32.mlir")
    op_tests_f32(name = "log_rank2_f32", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "log_rank2_f32.mlir")
    op_tests_f32(name = "log_rank3_f32", instances = ["(2,3,4)", "(10,20,30)", "(5,100,2)"], test = "log_rank3_f32.mlir")
    op_tests_f32(name = "log_rank4_f32", instances = ["(2,2,3,2)", "(2,3,4,50)", "(1,1,5,400)"], test = "log_rank4_f32.mlir")
    op_tests_f32(name = "log_plus_one_rank1_f32", instances = ["(8)", "(256)", "(450)"], test = "log_plus_one_rank1_f32.mlir")
    op_tests_f32(name = "log_plus_one_rank2_f32", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "log_plus_one_rank2_f32.mlir")
    op_tests_f32(name = "log_plus_one_rank3_f32", instances = ["(2,3,4)", "(10,20,30)", "(5,100,2)"], test = "log_plus_one_rank3_f32.mlir")
    op_tests_f32(name = "log_plus_one_rank4_f32", instances = ["(2,2,3,2)", "(2,3,4,50)", "(1,1,5,400)"], test = "log_plus_one_rank4_f32.mlir")
    op_tests_f32(name = "logistic_rank1_f32", instances = ["(8)", "(256)", "(450)"], test = "logistic_rank1_f32.mlir")
    op_tests_f32(name = "logistic_rank2_f32", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "logistic_rank2_f32.mlir")
    op_tests_f32(name = "logistic_rank3_f32", instances = ["(2,3,4)", "(10,20,30)", "(5,100,2)"], test = "logistic_rank3_f32.mlir")
    op_tests_f32(name = "logistic_rank4_f32", instances = ["(2,2,3,2)", "(2,3,4,50)", "(1,1,5,400)"], test = "logistic_rank4_f32.mlir")
    op_tests_f32(name = "rsqrt_rank1_f32", instances = ["(8)", "(256)", "(450)"], test = "rsqrt_rank1_f32.mlir")
    op_tests_f32(name = "rsqrt_rank2_f32", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "rsqrt_rank2_f32.mlir")
    op_tests_f32(name = "rsqrt_rank3_f32", instances = ["(2,3,4)", "(10,20,30)", "(5,100,2)"], test = "rsqrt_rank3_f32.mlir")
    op_tests_f32(name = "rsqrt_rank4_f32", instances = ["(2,2,3,2)", "(2,3,4,50)", "(1,1,5,400)"], test = "rsqrt_rank4_f32.mlir")
    op_tests_f32(name = "sine_rank1_f32", instances = ["(8)", "(256)", "(450)"], test = "sine_rank1_f32.mlir")
    op_tests_f32(name = "sine_rank2_f32", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "sine_rank2_f32.mlir")
    op_tests_f32(name = "sine_rank3_f32", instances = ["(2,3,4)", "(10,20,30)", "(5,100,2)"], test = "sine_rank3_f32.mlir")
    op_tests_f32(name = "sine_rank4_f32", instances = ["(2,2,3,2)", "(2,3,4,50)", "(1,1,5,400)"], test = "sine_rank4_f32.mlir")
    op_tests_f32(name = "sqrt_rank1_f32", instances = ["(8)", "(256)", "(450)"], test = "sqrt_rank1_f32.mlir")
    op_tests_f32(name = "sqrt_rank2_f32", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "sqrt_rank2_f32.mlir")
    op_tests_f32(name = "sqrt_rank3_f32", instances = ["(2,3,4)", "(10,20,30)", "(5,100,2)"], test = "sqrt_rank3_f32.mlir")
    op_tests_f32(name = "sqrt_rank4_f32", instances = ["(2,2,3,2)", "(2,3,4,50)", "(1,1,5,400)"], test = "sqrt_rank4_f32.mlir")

    # Reason: VMVX lacks math.tan legalization
    op_tests_f32(name = "tan_rank1_f32", instances = ["(8)", "(256)", "(450)"], test = "tan_rank1_f32.mlir", tags = ["manual"])

    # Reason: VMVX lacks math.tan legalization
    op_tests_f32(name = "tan_rank2_f32", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "tan_rank2_f32.mlir", tags = ["manual"])

    # Reason: VMVX lacks math.tan legalization
    op_tests_f32(name = "tan_rank3_f32", instances = ["(2,3,4)", "(10,20,30)", "(5,100,2)"], test = "tan_rank3_f32.mlir", tags = ["manual"])

    # Reason: VMVX lacks math.tan legalization
    op_tests_f32(name = "tan_rank4_f32", instances = ["(2,2,3,2)", "(2,3,4,50)", "(1,1,5,400)"], test = "tan_rank4_f32.mlir", tags = ["manual"])
    op_tests_f32(name = "tanh_rank1_f32", instances = ["(8)", "(256)", "(450)"], test = "tanh_rank1_f32.mlir")
    op_tests_f32(name = "tanh_rank2_f32", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "tanh_rank2_f32.mlir")
    op_tests_f32(name = "tanh_rank3_f32", instances = ["(2,3,4)", "(10,20,30)", "(5,100,2)"], test = "tanh_rank3_f32.mlir")
    op_tests_f32(name = "tanh_rank4_f32", instances = ["(2,2,3,2)", "(2,3,4,50)", "(1,1,5,400)"], test = "tanh_rank4_f32.mlir")

    # Reason: VMVX lacks math.copysign legalization for f32
    op_tests_f32(name = "sign_rank1_f32", instances = ["(8)", "(256)", "(450)"], test = "sign_rank1_f32.mlir", tags = ["manual"])

    # Reason: VMVX lacks math.copysign legalization for f32
    op_tests_f32(name = "sign_rank2_f32", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "sign_rank2_f32.mlir", tags = ["manual"])

    # Reason: VMVX lacks math.copysign legalization for f32
    op_tests_f32(name = "sign_rank3_f32", instances = ["(2,3,4)", "(10,20,30)", "(5,100,2)"], test = "sign_rank3_f32.mlir", tags = ["manual"])

    # Reason: VMVX lacks math.copysign legalization for f32
    op_tests_f32(name = "sign_rank4_f32", instances = ["(2,2,3,2)", "(2,3,4,50)", "(1,1,5,400)"], test = "sign_rank4_f32.mlir", tags = ["manual"])

    # Reason: CoralNPU atan2(0,0) returns NaN (spec expects 0)
    op_tests_f32(name = "atan2_rank1_f32", instances = ["(8)(8)", "(256)(256)", "(450)(450)"], test = "atan2_rank1_f32.mlir", tags = ["manual"])

    # Reason: CoralNPU atan2(0,0) returns NaN (spec expects 0)
    op_tests_f32(name = "atan2_rank2_f32", instances = ["(4,8)(4,8)", "(120,256)(120,256)", "(300,450)(300,450)"], test = "atan2_rank2_f32.mlir", tags = ["manual"])

    # Reason: CoralNPU atan2(0,0) returns NaN (spec expects 0)
    op_tests_f32(name = "atan2_rank3_f32", instances = ["(2,3,4)(2,3,4)", "(10,20,30)(10,20,30)", "(5,100,2)(5,100,2)"], test = "atan2_rank3_f32.mlir", tags = ["manual"])

    # Reason: CoralNPU atan2(0,0) returns NaN (spec expects 0)
    op_tests_f32(name = "atan2_rank4_f32", instances = ["(2,2,3,2)(2,2,3,2)", "(2,3,4,50)(2,3,4,50)", "(1,1,5,400)(1,1,5,400)"], test = "atan2_rank4_f32.mlir", tags = ["manual"])

    # Reason: CoralNPU pow(0,0) returns NaN (spec expects 1)
    op_tests_f32(name = "power_rank1_f32", instances = ["(8)(8)", "(256)(256)", "(450)(450)"], test = "power_rank1_f32.mlir", tags = ["manual"])

    # Reason: CoralNPU pow(0,0) returns NaN (spec expects 1)
    op_tests_f32(name = "power_rank2_f32", instances = ["(4,8)(4,8)", "(120,256)(120,256)", "(300,450)(300,450)"], test = "power_rank2_f32.mlir", tags = ["manual"])

    # Reason: CoralNPU pow(0,0) returns NaN (spec expects 1)
    op_tests_f32(name = "power_rank3_f32", instances = ["(2,3,4)(2,3,4)", "(10,20,30)(10,20,30)", "(5,100,2)(5,100,2)"], test = "power_rank3_f32.mlir", tags = ["manual"])

    # Reason: CoralNPU pow(0,0) returns NaN (spec expects 1)
    op_tests_f32(name = "power_rank4_f32", instances = ["(2,2,3,2)(2,2,3,2)", "(2,3,4,50)(2,3,4,50)", "(1,1,5,400)(1,1,5,400)"], test = "power_rank4_f32.mlir", tags = ["manual"])
    op_tests_f32(name = "remainder_rank1_f32", instances = ["(8)(8)", "(256)(256)", "(450)(450)"], test = "remainder_rank1_f32.mlir")
    op_tests_f32(name = "remainder_rank2_f32", instances = ["(4,8)(4,8)", "(120,256)(120,256)", "(300,450)(300,450)"], test = "remainder_rank2_f32.mlir")
    op_tests_f32(name = "remainder_rank3_f32", instances = ["(2,3,4)(2,3,4)", "(10,20,30)(10,20,30)", "(5,100,2)(5,100,2)"], test = "remainder_rank3_f32.mlir")
    op_tests_f32(name = "remainder_rank4_f32", instances = ["(2,2,3,2)(2,2,3,2)", "(2,3,4,50)(2,3,4,50)", "(1,1,5,400)(1,1,5,400)"], test = "remainder_rank4_f32.mlir")
    op_tests_f32(name = "is_finite_rank1_f32", instances = ["(8)", "(256)", "(450)"], test = "is_finite_rank1_f32.mlir")
    op_tests_f32(name = "is_finite_rank2_f32", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "is_finite_rank2_f32.mlir")
    op_tests_f32(name = "is_finite_rank3_f32", instances = ["(2,3,4)", "(10,20,30)", "(5,100,2)"], test = "is_finite_rank3_f32.mlir")
    op_tests_f32(name = "is_finite_rank4_f32", instances = ["(2,2,3,2)", "(2,3,4,50)", "(1,1,5,400)"], test = "is_finite_rank4_f32.mlir")
