# Generated from op_tests_f32.bzl. Do not edit directly.

op_tests(
  NAME "abs_rank1_f32"
  TEST "abs_rank1_f32.mlir"
  INSTANCES
    "(8)"
    "(256)"
    "(450)"
)

op_tests(
  NAME "abs_rank2_f32"
  TEST "abs_rank2_f32.mlir"
  INSTANCES
    "(4,8)"
    "(120,256)"
    "(300,450)"
)

op_tests(
  NAME "abs_rank3_f32"
  TEST "abs_rank3_f32.mlir"
  INSTANCES
    "(2,3,4)"
    "(10,20,30)"
    "(5,100,2)"
)

op_tests(
  NAME "abs_rank4_f32"
  TEST "abs_rank4_f32.mlir"
  INSTANCES
    "(2,2,3,2)"
    "(2,3,4,50)"
    "(1,1,5,400)"
)

op_tests(
  NAME "add_rank1_f32"
  TEST "add_rank1_f32.mlir"
  INSTANCES
    "(8)(8)"
    "(256)(256)"
    "(450)(450)"
)

op_tests(
  NAME "add_rank2_f32"
  TEST "add_rank2_f32.mlir"
  INSTANCES
    "(4,8)(4,8)"
    "(120,256)(120,256)"
    "(300,450)(300,450)"
)

op_tests(
  NAME "add_rank3_f32"
  TEST "add_rank3_f32.mlir"
  INSTANCES
    "(2,3,4)(2,3,4)"
    "(10,20,30)(10,20,30)"
    "(5,100,2)(5,100,2)"
)

op_tests(
  NAME "add_rank4_f32"
  TEST "add_rank4_f32.mlir"
  INSTANCES
    "(2,2,3,2)(2,2,3,2)"
    "(2,3,4,50)(2,3,4,50)"
    "(1,1,5,400)(1,1,5,400)"
)

op_tests(
  NAME "broadcast_in_dim_rank2_f32_256"
  TEST "broadcast_in_dim_rank2_f32_256.mlir"
  INSTANCES
    "(256)"
)

op_tests(
  NAME "broadcast_in_dim_rank2_f32_450"
  TEST "broadcast_in_dim_rank2_f32_450.mlir"
  INSTANCES
    "(450)"
)

op_tests(
  NAME "broadcast_in_dim_rank2_f32_8"
  TEST "broadcast_in_dim_rank2_f32_8.mlir"
  INSTANCES
    "(8)"
)

op_tests(
  NAME "broadcast_in_dim_rank3_f32_120_256"
  TEST "broadcast_in_dim_rank3_f32_120_256.mlir"
  INSTANCES
    "(120,256)"
)

op_tests(
  NAME "broadcast_in_dim_rank3_f32_300_450"
  TEST "broadcast_in_dim_rank3_f32_300_450.mlir"
  INSTANCES
    "(300,450)"
)

op_tests(
  NAME "broadcast_in_dim_rank3_f32_4_8"
  TEST "broadcast_in_dim_rank3_f32_4_8.mlir"
  INSTANCES
    "(4,8)"
)

op_tests(
  NAME "broadcast_in_dim_rank4_f32_10_20_30"
  TEST "broadcast_in_dim_rank4_f32_10_20_30.mlir"
  INSTANCES
    "(10,20,30)"
)

op_tests(
  NAME "broadcast_in_dim_rank4_f32_2_3_4"
  TEST "broadcast_in_dim_rank4_f32_2_3_4.mlir"
  INSTANCES
    "(2,3,4)"
)

op_tests(
  NAME "broadcast_in_dim_rank4_f32_5_100_2"
  TEST "broadcast_in_dim_rank4_f32_5_100_2.mlir"
  INSTANCES
    "(5,100,2)"
)

op_tests(
  NAME "compare_eq_rank1_f32"
  TEST "compare_eq_rank1_f32.mlir"
  INSTANCES
    "(8)(8)"
    "(256)(256)"
    "(450)(450)"
)

op_tests(
  NAME "compare_eq_rank2_f32"
  TEST "compare_eq_rank2_f32.mlir"
  INSTANCES
    "(4,8)(4,8)"
    "(120,256)(120,256)"
    "(300,450)(300,450)"
)

op_tests(
  NAME "compare_eq_rank3_f32"
  TEST "compare_eq_rank3_f32.mlir"
  INSTANCES
    "(2,3,4)(2,3,4)"
    "(10,20,30)(10,20,30)"
    "(5,100,2)(5,100,2)"
)

op_tests(
  NAME "compare_eq_rank4_f32"
  TEST "compare_eq_rank4_f32.mlir"
  INSTANCES
    "(2,2,3,2)(2,2,3,2)"
    "(2,3,4,50)(2,3,4,50)"
    "(1,1,5,400)(1,1,5,400)"
)

op_tests(
  NAME "concatenate_rank1_f32"
  TEST "concatenate_rank1_f32.mlir"
  INSTANCES
    "(8)(16)"
    "(256)(100)"
    "(450)(50)"
)

op_tests(
  NAME "concatenate_rank2_f32"
  TEST "concatenate_rank2_f32.mlir"
  INSTANCES
    "(4,8)(4,16)"
    "(120,256)(120,100)"
    "(300,450)(300,50)"
)

op_tests(
  NAME "concatenate_rank3_f32"
  TEST "concatenate_rank3_f32.mlir"
  INSTANCES
    "(2,3,4)(2,3,8)"
    "(10,20,30)(10,20,10)"
    "(5,100,2)(5,100,5)"
)

op_tests(
  NAME "concatenate_rank4_f32"
  TEST "concatenate_rank4_f32.mlir"
  INSTANCES
    "(2,2,3,2)(2,2,3,4)"
    "(2,3,4,50)(2,3,4,10)"
    "(1,1,5,400)(1,1,5,100)"
)

op_tests(
  NAME "convolution_rank2_f32"
  TEST "convolution_rank2_f32.mlir"
  INSTANCES
    "(4,8)(8,4)"
    "(120,256)(256,300)"
    "(300,100)(100,450)"
)

op_tests(
  NAME "convolution_rank3_f32"
  TEST "convolution_rank3_f32.mlir"
  INSTANCES
    "(1,8,3)(3,3,16)"
    "(2,120,3)(5,3,32)"
    "(1,300,3)(3,3,8)"
)

op_tests(
  NAME "convolution_rank4_f32"
  TEST "convolution_rank4_f32.mlir"
  INSTANCES
    "(1,8,8,3)(3,3,3,16)"
)

op_tests(
  NAME "divide_rank1_f32"
  TEST "divide_rank1_f32.mlir"
  INSTANCES
    "(8)(8)"
    "(256)(256)"
    "(450)(450)"
)

op_tests(
  NAME "divide_rank2_f32"
  TEST "divide_rank2_f32.mlir"
  INSTANCES
    "(4,8)(4,8)"
    "(120,256)(120,256)"
    "(300,450)(300,450)"
)

op_tests(
  NAME "divide_rank3_f32"
  TEST "divide_rank3_f32.mlir"
  INSTANCES
    "(2,3,4)(2,3,4)"
    "(10,20,30)(10,20,30)"
    "(5,100,2)(5,100,2)"
)

op_tests(
  NAME "divide_rank4_f32"
  TEST "divide_rank4_f32.mlir"
  INSTANCES
    "(2,2,3,2)(2,2,3,2)"
    "(2,3,4,50)(2,3,4,50)"
    "(1,1,5,400)(1,1,5,400)"
)

op_tests(
  NAME "dynamic_broadcast_in_dim_rank2_f32"
  TEST "dynamic_broadcast_in_dim_rank2_f32.mlir"
  INSTANCES
    "(8)"
    "(256)"
    "(450)"
)

op_tests(
  NAME "dynamic_broadcast_in_dim_rank3_f32"
  TEST "dynamic_broadcast_in_dim_rank3_f32.mlir"
  INSTANCES
    "(4,8)"
    "(120,256)"
    "(300,450)"
)

op_tests(
  NAME "dynamic_broadcast_in_dim_rank4_f32"
  TEST "dynamic_broadcast_in_dim_rank4_f32.mlir"
  INSTANCES
    "(2,3,4)"
    "(10,20,30)"
    "(5,100,2)"
)

op_tests(
  NAME "matmul_rank1_f32"
  TEST "matmul_rank1_f32.mlir"
  INSTANCES
    "(8)(8)"
    "(256)(256)"
)

op_tests(
  NAME "matmul_rank2_f32"
  TEST "matmul_rank2_f32.mlir"
  INSTANCES
    "(4,8)(8,4)"
    "(120,256)(256,300)"
    "(300,100)(100,450)"
)

op_tests(
  NAME "maximum_rank1_f32"
  TEST "maximum_rank1_f32.mlir"
  INSTANCES
    "(8)(8)"
    "(256)(256)"
    "(450)(450)"
)

op_tests(
  NAME "maximum_rank2_f32"
  TEST "maximum_rank2_f32.mlir"
  INSTANCES
    "(4,8)(4,8)"
    "(120,256)(120,256)"
    "(300,450)(300,450)"
)

op_tests(
  NAME "maximum_rank3_f32"
  TEST "maximum_rank3_f32.mlir"
  INSTANCES
    "(2,3,4)(2,3,4)"
    "(10,20,30)(10,20,30)"
    "(5,100,2)(5,100,2)"
)

op_tests(
  NAME "maximum_rank4_f32"
  TEST "maximum_rank4_f32.mlir"
  INSTANCES
    "(2,2,3,2)(2,2,3,2)"
    "(2,3,4,50)(2,3,4,50)"
    "(1,1,5,400)(1,1,5,400)"
)

op_tests(
  NAME "minimum_rank1_f32"
  TEST "minimum_rank1_f32.mlir"
  INSTANCES
    "(8)(8)"
    "(256)(256)"
    "(450)(450)"
)

op_tests(
  NAME "minimum_rank2_f32"
  TEST "minimum_rank2_f32.mlir"
  INSTANCES
    "(4,8)(4,8)"
    "(120,256)(120,256)"
    "(300,450)(300,450)"
)

op_tests(
  NAME "minimum_rank3_f32"
  TEST "minimum_rank3_f32.mlir"
  INSTANCES
    "(2,3,4)(2,3,4)"
    "(10,20,30)(10,20,30)"
    "(5,100,2)(5,100,2)"
)

op_tests(
  NAME "minimum_rank4_f32"
  TEST "minimum_rank4_f32.mlir"
  INSTANCES
    "(2,2,3,2)(2,2,3,2)"
    "(2,3,4,50)(2,3,4,50)"
    "(1,1,5,400)(1,1,5,400)"
)

op_tests(
  NAME "multiply_rank1_f32"
  TEST "multiply_rank1_f32.mlir"
  INSTANCES
    "(8)(8)"
    "(256)(256)"
    "(450)(450)"
)

op_tests(
  NAME "multiply_rank2_f32"
  TEST "multiply_rank2_f32.mlir"
  INSTANCES
    "(4,8)(4,8)"
    "(120,256)(120,256)"
    "(300,450)(300,450)"
)

op_tests(
  NAME "multiply_rank3_f32"
  TEST "multiply_rank3_f32.mlir"
  INSTANCES
    "(2,3,4)(2,3,4)"
    "(10,20,30)(10,20,30)"
    "(5,100,2)(5,100,2)"
)

op_tests(
  NAME "multiply_rank4_f32"
  TEST "multiply_rank4_f32.mlir"
  INSTANCES
    "(2,2,3,2)(2,2,3,2)"
    "(2,3,4,50)(2,3,4,50)"
    "(1,1,5,400)(1,1,5,400)"
)

op_tests(
  NAME "negate_rank1_f32"
  TEST "negate_rank1_f32.mlir"
  INSTANCES
    "(8)"
    "(256)"
    "(450)"
)

op_tests(
  NAME "negate_rank2_f32"
  TEST "negate_rank2_f32.mlir"
  INSTANCES
    "(4,8)"
    "(120,256)"
    "(300,450)"
)

op_tests(
  NAME "negate_rank3_f32"
  TEST "negate_rank3_f32.mlir"
  INSTANCES
    "(2,3,4)"
    "(10,20,30)"
    "(5,100,2)"
)

op_tests(
  NAME "negate_rank4_f32"
  TEST "negate_rank4_f32.mlir"
  INSTANCES
    "(2,2,3,2)"
    "(2,3,4,50)"
    "(1,1,5,400)"
)

op_tests(
  NAME "reduce_sum_rank1_f32"
  TEST "reduce_sum_rank1_f32.mlir"
  INSTANCES
    "(8)"
    "(256)"
    "(450)"
)

op_tests(
  NAME "reduce_sum_rank2_f32"
  TEST "reduce_sum_rank2_f32.mlir"
  INSTANCES
    "(4,8)"
    "(120,256)"
    "(300,450)"
)

op_tests(
  NAME "reduce_sum_rank3_f32"
  TEST "reduce_sum_rank3_f32.mlir"
  INSTANCES
    "(2,3,4)"
    "(10,20,30)"
    "(5,100,2)"
)

op_tests(
  NAME "reduce_sum_rank4_f32"
  TEST "reduce_sum_rank4_f32.mlir"
  INSTANCES
    "(2,2,3,2)"
    "(2,3,4,50)"
    "(1,1,5,400)"
)

op_tests(
  NAME "select_rank1_f32"
  TEST "select_rank1_f32.mlir"
  INSTANCES
    "(8)(8)(8)"
    "(256)(256)(256)"
    "(450)(450)(450)"
)

op_tests(
  NAME "select_rank2_f32"
  TEST "select_rank2_f32.mlir"
  INSTANCES
    "(4,8)(4,8)(4,8)"
    "(120,256)(120,256)(120,256)"
    "(300,450)(300,450)(300,450)"
)

op_tests(
  NAME "select_rank3_f32"
  TEST "select_rank3_f32.mlir"
  INSTANCES
    "(2,3,4)(2,3,4)(2,3,4)"
    "(10,20,30)(10,20,30)(10,20,30)"
    "(5,100,2)(5,100,2)(5,100,2)"
)

op_tests(
  NAME "select_rank4_f32"
  TEST "select_rank4_f32.mlir"
  INSTANCES
    "(2,2,3,2)(2,2,3,2)(2,2,3,2)"
    "(2,3,4,50)(2,3,4,50)(2,3,4,50)"
    "(1,1,5,400)(1,1,5,400)(1,1,5,400)"
)

op_tests(
  NAME "slice_rank1_f32"
  TEST "slice_rank1_f32.mlir"
  INSTANCES
    "(8)"
    "(256)"
    "(450)"
)

op_tests(
  NAME "slice_rank2_f32"
  TEST "slice_rank2_f32.mlir"
  INSTANCES
    "(4,8)"
    "(120,256)"
    "(300,450)"
)

op_tests(
  NAME "slice_rank3_f32"
  TEST "slice_rank3_f32.mlir"
  INSTANCES
    "(2,3,4)"
    "(10,20,30)"
    "(5,100,4)"
)

op_tests(
  NAME "slice_rank4_f32"
  TEST "slice_rank4_f32.mlir"
  INSTANCES
    "(2,2,3,4)"
    "(2,3,4,50)"
    "(2,2,5,400)"
)

op_tests(
  NAME "subtract_rank1_f32"
  TEST "subtract_rank1_f32.mlir"
  INSTANCES
    "(8)(8)"
    "(256)(256)"
    "(450)(450)"
)

op_tests(
  NAME "subtract_rank2_f32"
  TEST "subtract_rank2_f32.mlir"
  INSTANCES
    "(4,8)(4,8)"
    "(120,256)(120,256)"
    "(300,450)(300,450)"
)

op_tests(
  NAME "subtract_rank3_f32"
  TEST "subtract_rank3_f32.mlir"
  INSTANCES
    "(2,3,4)(2,3,4)"
    "(10,20,30)(10,20,30)"
    "(5,100,2)(5,100,2)"
)

op_tests(
  NAME "subtract_rank4_f32"
  TEST "subtract_rank4_f32.mlir"
  INSTANCES
    "(2,2,3,2)(2,2,3,2)"
    "(2,3,4,50)(2,3,4,50)"
    "(1,1,5,400)(1,1,5,400)"
)

op_tests(
  NAME "transpose_rank2_f32"
  TEST "transpose_rank2_f32.mlir"
  INSTANCES
    "(4,8)"
    "(120,256)"
    "(300,450)"
)

op_tests(
  NAME "transpose_rank3_f32"
  TEST "transpose_rank3_f32.mlir"
  INSTANCES
    "(2,3,4)"
    "(10,20,30)"
    "(5,100,2)"
)

op_tests(
  NAME "transpose_rank4_f32"
  TEST "transpose_rank4_f32.mlir"
  INSTANCES
    "(2,2,3,2)"
    "(2,3,4,50)"
    "(1,1,5,400)"
)

op_tests(
  NAME "pad_rank2_f32"
  TEST "pad_rank2_f32.mlir"
  INSTANCES
    "(4,8)"
    "(120,256)"
    "(300,450)"
)

op_tests(
  NAME "clamp_rank2_f32"
  TEST "clamp_rank2_f32.mlir"
  INSTANCES
    "(4,8)"
    "(120,256)"
    "(300,450)"
)

op_tests(
  NAME "broadcast_rank2_f32"
  TEST "broadcast_rank2_f32.mlir"
  INSTANCES
    "(8)"
    "(256)"
    "(450)"
)

op_tests(
  NAME "if_rank1_f32"
  TEST "if_rank1_f32.mlir"
  INSTANCES
    "(1)(8)"
    "(1)(256)"
)

op_tests(
  NAME "case_rank1_f32"
  TEST "case_rank1_f32.mlir"
  INSTANCES
    "(1)(8)"
    "(1)(256)"
)

op_tests(
  NAME "dot_general_rank3_f32"
  TEST "dot_general_rank3_f32.mlir"
  INSTANCES
    "(2,2,3)(2,3,4)"
    "(1,8,16)(1,16,8)"
)

op_tests(
  NAME "gather_rank1_f32"
  TEST "gather_rank1_f32.mlir"
  INSTANCES
    "(10)(8)"
    "(300)(256)"
)

op_tests(
  NAME "scatter_rank1_f32"
  TEST "scatter_rank1_f32.mlir"
  INSTANCES
    "(10)(3)(3)"
    "(300)(256)(256)"
)

op_tests(
  NAME "sort_rank1_f32"
  TEST "sort_rank1_f32.mlir"
  INSTANCES
    "(8)"
    "(256)"
)

op_tests(
  NAME "tuple_rank1_f32"
  TEST "tuple_rank1_f32.mlir"
  INSTANCES
    "(8)(8)"
    "(256)(256)"
)

op_tests(
  NAME "bitcast_convert_rank1_f32_i32"
  TEST "bitcast_convert_rank1_f32_i32.mlir"
  INSTANCES
    "(8)"
    "(256)"
)

op_tests(
  NAME "batch_norm_inference_rank4_f32"
  TEST "batch_norm_inference_rank4_f32.mlir"
  INSTANCES
    "(1,2,2,3)(3)(3)(3)(3)"
    "(1,8,16,32)(32)(32)(32)(32)"
)

op_tests(
  NAME "ceil_rank1_f32"
  TEST "ceil_rank1_f32.mlir"
  INSTANCES
    "(8)"
    "(256)"
    "(450)"
)

op_tests(
  NAME "ceil_rank2_f32"
  TEST "ceil_rank2_f32.mlir"
  INSTANCES
    "(4,8)"
    "(120,256)"
    "(300,450)"
)

op_tests(
  NAME "ceil_rank3_f32"
  TEST "ceil_rank3_f32.mlir"
  INSTANCES
    "(2,3,4)"
    "(10,20,30)"
    "(5,100,2)"
)

op_tests(
  NAME "ceil_rank4_f32"
  TEST "ceil_rank4_f32.mlir"
  INSTANCES
    "(2,2,3,2)"
    "(2,3,4,50)"
    "(1,1,5,400)"
)

op_tests(
  NAME "cosine_rank1_f32"
  TEST "cosine_rank1_f32.mlir"
  INSTANCES
    "(8)"
    "(256)"
    "(450)"
)

op_tests(
  NAME "cosine_rank2_f32"
  TEST "cosine_rank2_f32.mlir"
  INSTANCES
    "(4,8)"
    "(120,256)"
    "(300,450)"
)

op_tests(
  NAME "cosine_rank3_f32"
  TEST "cosine_rank3_f32.mlir"
  INSTANCES
    "(2,3,4)"
    "(10,20,30)"
    "(5,100,2)"
)

op_tests(
  NAME "cosine_rank4_f32"
  TEST "cosine_rank4_f32.mlir"
  INSTANCES
    "(2,2,3,2)"
    "(2,3,4,50)"
    "(1,1,5,400)"
)

op_tests(
  NAME "exponential_rank1_f32"
  TEST "exponential_rank1_f32.mlir"
  INSTANCES
    "(8)"
    "(256)"
    "(450)"
)

op_tests(
  NAME "exponential_rank2_f32"
  TEST "exponential_rank2_f32.mlir"
  INSTANCES
    "(4,8)"
    "(120,256)"
    "(300,450)"
)

op_tests(
  NAME "exponential_rank3_f32"
  TEST "exponential_rank3_f32.mlir"
  INSTANCES
    "(2,3,4)"
    "(10,20,30)"
    "(5,100,2)"
)

op_tests(
  NAME "exponential_rank4_f32"
  TEST "exponential_rank4_f32.mlir"
  INSTANCES
    "(2,2,3,2)"
    "(2,3,4,50)"
    "(1,1,5,400)"
)

op_tests(
  NAME "exponential_minus_one_rank1_f32"
  TEST "exponential_minus_one_rank1_f32.mlir"
  INSTANCES
    "(8)"
    "(256)"
    "(450)"
)

op_tests(
  NAME "exponential_minus_one_rank2_f32"
  TEST "exponential_minus_one_rank2_f32.mlir"
  INSTANCES
    "(4,8)"
    "(120,256)"
    "(300,450)"
)

op_tests(
  NAME "exponential_minus_one_rank3_f32"
  TEST "exponential_minus_one_rank3_f32.mlir"
  INSTANCES
    "(2,3,4)"
    "(10,20,30)"
    "(5,100,2)"
)

op_tests(
  NAME "exponential_minus_one_rank4_f32"
  TEST "exponential_minus_one_rank4_f32.mlir"
  INSTANCES
    "(2,2,3,2)"
    "(2,3,4,50)"
    "(1,1,5,400)"
)

op_tests(
  NAME "floor_rank1_f32"
  TEST "floor_rank1_f32.mlir"
  INSTANCES
    "(8)"
    "(256)"
    "(450)"
)

op_tests(
  NAME "floor_rank2_f32"
  TEST "floor_rank2_f32.mlir"
  INSTANCES
    "(4,8)"
    "(120,256)"
    "(300,450)"
)

op_tests(
  NAME "floor_rank3_f32"
  TEST "floor_rank3_f32.mlir"
  INSTANCES
    "(2,3,4)"
    "(10,20,30)"
    "(5,100,2)"
)

op_tests(
  NAME "floor_rank4_f32"
  TEST "floor_rank4_f32.mlir"
  INSTANCES
    "(2,2,3,2)"
    "(2,3,4,50)"
    "(1,1,5,400)"
)

op_tests(
  NAME "log_rank1_f32"
  TEST "log_rank1_f32.mlir"
  INSTANCES
    "(8)"
    "(256)"
    "(450)"
)

op_tests(
  NAME "log_rank2_f32"
  TEST "log_rank2_f32.mlir"
  INSTANCES
    "(4,8)"
    "(120,256)"
    "(300,450)"
)

op_tests(
  NAME "log_rank3_f32"
  TEST "log_rank3_f32.mlir"
  INSTANCES
    "(2,3,4)"
    "(10,20,30)"
    "(5,100,2)"
)

op_tests(
  NAME "log_rank4_f32"
  TEST "log_rank4_f32.mlir"
  INSTANCES
    "(2,2,3,2)"
    "(2,3,4,50)"
    "(1,1,5,400)"
)

op_tests(
  NAME "log_plus_one_rank1_f32"
  TEST "log_plus_one_rank1_f32.mlir"
  INSTANCES
    "(8)"
    "(256)"
    "(450)"
)

op_tests(
  NAME "log_plus_one_rank2_f32"
  TEST "log_plus_one_rank2_f32.mlir"
  INSTANCES
    "(4,8)"
    "(120,256)"
    "(300,450)"
)

op_tests(
  NAME "log_plus_one_rank3_f32"
  TEST "log_plus_one_rank3_f32.mlir"
  INSTANCES
    "(2,3,4)"
    "(10,20,30)"
    "(5,100,2)"
)

op_tests(
  NAME "log_plus_one_rank4_f32"
  TEST "log_plus_one_rank4_f32.mlir"
  INSTANCES
    "(2,2,3,2)"
    "(2,3,4,50)"
    "(1,1,5,400)"
)

op_tests(
  NAME "logistic_rank1_f32"
  TEST "logistic_rank1_f32.mlir"
  INSTANCES
    "(8)"
    "(256)"
    "(450)"
)

op_tests(
  NAME "logistic_rank2_f32"
  TEST "logistic_rank2_f32.mlir"
  INSTANCES
    "(4,8)"
    "(120,256)"
    "(300,450)"
)

op_tests(
  NAME "logistic_rank3_f32"
  TEST "logistic_rank3_f32.mlir"
  INSTANCES
    "(2,3,4)"
    "(10,20,30)"
    "(5,100,2)"
)

op_tests(
  NAME "logistic_rank4_f32"
  TEST "logistic_rank4_f32.mlir"
  INSTANCES
    "(2,2,3,2)"
    "(2,3,4,50)"
    "(1,1,5,400)"
)

op_tests(
  NAME "rsqrt_rank1_f32"
  TEST "rsqrt_rank1_f32.mlir"
  INSTANCES
    "(8)"
    "(256)"
    "(450)"
)

op_tests(
  NAME "rsqrt_rank2_f32"
  TEST "rsqrt_rank2_f32.mlir"
  INSTANCES
    "(4,8)"
    "(120,256)"
    "(300,450)"
)

op_tests(
  NAME "rsqrt_rank3_f32"
  TEST "rsqrt_rank3_f32.mlir"
  INSTANCES
    "(2,3,4)"
    "(10,20,30)"
    "(5,100,2)"
)

op_tests(
  NAME "rsqrt_rank4_f32"
  TEST "rsqrt_rank4_f32.mlir"
  INSTANCES
    "(2,2,3,2)"
    "(2,3,4,50)"
    "(1,1,5,400)"
)

op_tests(
  NAME "sine_rank1_f32"
  TEST "sine_rank1_f32.mlir"
  INSTANCES
    "(8)"
    "(256)"
    "(450)"
)

op_tests(
  NAME "sine_rank2_f32"
  TEST "sine_rank2_f32.mlir"
  INSTANCES
    "(4,8)"
    "(120,256)"
    "(300,450)"
)

op_tests(
  NAME "sine_rank3_f32"
  TEST "sine_rank3_f32.mlir"
  INSTANCES
    "(2,3,4)"
    "(10,20,30)"
    "(5,100,2)"
)

op_tests(
  NAME "sine_rank4_f32"
  TEST "sine_rank4_f32.mlir"
  INSTANCES
    "(2,2,3,2)"
    "(2,3,4,50)"
    "(1,1,5,400)"
)

op_tests(
  NAME "sqrt_rank1_f32"
  TEST "sqrt_rank1_f32.mlir"
  INSTANCES
    "(8)"
    "(256)"
    "(450)"
)

op_tests(
  NAME "sqrt_rank2_f32"
  TEST "sqrt_rank2_f32.mlir"
  INSTANCES
    "(4,8)"
    "(120,256)"
    "(300,450)"
)

op_tests(
  NAME "sqrt_rank3_f32"
  TEST "sqrt_rank3_f32.mlir"
  INSTANCES
    "(2,3,4)"
    "(10,20,30)"
    "(5,100,2)"
)

op_tests(
  NAME "sqrt_rank4_f32"
  TEST "sqrt_rank4_f32.mlir"
  INSTANCES
    "(2,2,3,2)"
    "(2,3,4,50)"
    "(1,1,5,400)"
)

op_tests(
  NAME "tanh_rank1_f32"
  TEST "tanh_rank1_f32.mlir"
  INSTANCES
    "(8)"
    "(256)"
    "(450)"
)

op_tests(
  NAME "tanh_rank2_f32"
  TEST "tanh_rank2_f32.mlir"
  INSTANCES
    "(4,8)"
    "(120,256)"
    "(300,450)"
)

op_tests(
  NAME "tanh_rank3_f32"
  TEST "tanh_rank3_f32.mlir"
  INSTANCES
    "(2,3,4)"
    "(10,20,30)"
    "(5,100,2)"
)

op_tests(
  NAME "tanh_rank4_f32"
  TEST "tanh_rank4_f32.mlir"
  INSTANCES
    "(2,2,3,2)"
    "(2,3,4,50)"
    "(1,1,5,400)"
)

op_tests(
  NAME "remainder_rank1_f32"
  TEST "remainder_rank1_f32.mlir"
  INSTANCES
    "(8)(8)"
    "(256)(256)"
    "(450)(450)"
)

op_tests(
  NAME "remainder_rank2_f32"
  TEST "remainder_rank2_f32.mlir"
  INSTANCES
    "(4,8)(4,8)"
    "(120,256)(120,256)"
    "(300,450)(300,450)"
)

op_tests(
  NAME "remainder_rank3_f32"
  TEST "remainder_rank3_f32.mlir"
  INSTANCES
    "(2,3,4)(2,3,4)"
    "(10,20,30)(10,20,30)"
    "(5,100,2)(5,100,2)"
)

op_tests(
  NAME "remainder_rank4_f32"
  TEST "remainder_rank4_f32.mlir"
  INSTANCES
    "(2,2,3,2)(2,2,3,2)"
    "(2,3,4,50)(2,3,4,50)"
    "(1,1,5,400)(1,1,5,400)"
)

op_tests(
  NAME "is_finite_rank1_f32"
  TEST "is_finite_rank1_f32.mlir"
  INSTANCES
    "(8)"
    "(256)"
    "(450)"
)

op_tests(
  NAME "is_finite_rank2_f32"
  TEST "is_finite_rank2_f32.mlir"
  INSTANCES
    "(4,8)"
    "(120,256)"
    "(300,450)"
)

op_tests(
  NAME "is_finite_rank3_f32"
  TEST "is_finite_rank3_f32.mlir"
  INSTANCES
    "(2,3,4)"
    "(10,20,30)"
    "(5,100,2)"
)

op_tests(
  NAME "is_finite_rank4_f32"
  TEST "is_finite_rank4_f32.mlir"
  INSTANCES
    "(2,2,3,2)"
    "(2,3,4,50)"
    "(1,1,5,400)"
)

