# Generated from op_tests_i8.bzl. Do not edit directly.

op_tests(
  NAME "abs_rank1_i8"
  TEST "abs_rank1_i8.mlir"
  INSTANCES
    "(8)"
    "(256)"
    "(450)"
)

op_tests(
  NAME "abs_rank2_i8"
  TEST "abs_rank2_i8.mlir"
  INSTANCES
    "(4,8)"
    "(120,256)"
    "(300,450)"
)

op_tests(
  NAME "abs_rank3_i8"
  TEST "abs_rank3_i8.mlir"
  INSTANCES
    "(2,3,4)"
    "(10,20,30)"
    "(5,100,2)"
)

op_tests(
  NAME "abs_rank4_i8"
  TEST "abs_rank4_i8.mlir"
  INSTANCES
    "(2,2,3,2)"
    "(2,3,4,50)"
    "(1,1,5,400)"
)

op_tests(
  NAME "add_rank1_i8"
  TEST "add_rank1_i8.mlir"
  INSTANCES
    "(8)(8)"
    "(256)(256)"
    "(450)(450)"
)

op_tests(
  NAME "add_rank2_i8"
  TEST "add_rank2_i8.mlir"
  INSTANCES
    "(4,8)(4,8)"
    "(120,256)(120,256)"
    "(300,450)(300,450)"
)

op_tests(
  NAME "add_rank3_i8"
  TEST "add_rank3_i8.mlir"
  INSTANCES
    "(2,3,4)(2,3,4)"
    "(10,20,30)(10,20,30)"
    "(5,100,2)(5,100,2)"
)

op_tests(
  NAME "add_rank4_i8"
  TEST "add_rank4_i8.mlir"
  INSTANCES
    "(2,2,3,2)(2,2,3,2)"
    "(2,3,4,50)(2,3,4,50)"
    "(1,1,5,400)(1,1,5,400)"
)

op_tests(
  NAME "broadcast_in_dim_rank2_i8_256"
  TEST "broadcast_in_dim_rank2_i8_256.mlir"
  INSTANCES
    "(256)"
)

op_tests(
  NAME "broadcast_in_dim_rank2_i8_450"
  TEST "broadcast_in_dim_rank2_i8_450.mlir"
  INSTANCES
    "(450)"
)

op_tests(
  NAME "broadcast_in_dim_rank2_i8_8"
  TEST "broadcast_in_dim_rank2_i8_8.mlir"
  INSTANCES
    "(8)"
)

op_tests(
  NAME "broadcast_in_dim_rank3_i8_120_256"
  TEST "broadcast_in_dim_rank3_i8_120_256.mlir"
  INSTANCES
    "(120,256)"
)

op_tests(
  NAME "broadcast_in_dim_rank3_i8_300_450"
  TEST "broadcast_in_dim_rank3_i8_300_450.mlir"
  INSTANCES
    "(300,450), [manual]"
)

op_tests(
  NAME "broadcast_in_dim_rank3_i8_4_8"
  TEST "broadcast_in_dim_rank3_i8_4_8.mlir"
  INSTANCES
    "(4,8)"
)

op_tests(
  NAME "broadcast_in_dim_rank4_i8_10_20_30"
  TEST "broadcast_in_dim_rank4_i8_10_20_30.mlir"
  INSTANCES
    "(10,20,30)"
)

op_tests(
  NAME "broadcast_in_dim_rank4_i8_2_3_4"
  TEST "broadcast_in_dim_rank4_i8_2_3_4.mlir"
  INSTANCES
    "(2,3,4)"
)

op_tests(
  NAME "broadcast_in_dim_rank4_i8_5_100_2"
  TEST "broadcast_in_dim_rank4_i8_5_100_2.mlir"
  INSTANCES
    "(5,100,2)"
)

op_tests(
  NAME "compare_eq_rank1_i8"
  TEST "compare_eq_rank1_i8.mlir"
  INSTANCES
    "(8)(8)"
    "(256)(256)"
    "(450)(450)"
)

op_tests(
  NAME "compare_eq_rank2_i8"
  TEST "compare_eq_rank2_i8.mlir"
  INSTANCES
    "(4,8)(4,8)"
    "(120,256)(120,256)"
    "(300,450)(300,450)"
)

op_tests(
  NAME "compare_eq_rank3_i8"
  TEST "compare_eq_rank3_i8.mlir"
  INSTANCES
    "(2,3,4)(2,3,4)"
    "(10,20,30)(10,20,30)"
    "(5,100,2)(5,100,2)"
)

op_tests(
  NAME "compare_eq_rank4_i8"
  TEST "compare_eq_rank4_i8.mlir"
  INSTANCES
    "(2,2,3,2)(2,2,3,2)"
    "(2,3,4,50)(2,3,4,50)"
    "(1,1,5,400)(1,1,5,400)"
)

op_tests(
  NAME "concatenate_rank1_i8"
  TEST "concatenate_rank1_i8.mlir"
  INSTANCES
    "(8)(16)"
    "(256)(100)"
    "(450)(50)"
)

op_tests(
  NAME "concatenate_rank2_i8"
  TEST "concatenate_rank2_i8.mlir"
  INSTANCES
    "(4,8)(4,16)"
    "(120,256)(120,100)"
    "(300,450)(300,50)"
)

op_tests(
  NAME "concatenate_rank3_i8"
  TEST "concatenate_rank3_i8.mlir"
  INSTANCES
    "(2,3,4)(2,3,8)"
    "(10,20,30)(10,20,10)"
    "(5,100,2)(5,100,5)"
)

op_tests(
  NAME "concatenate_rank4_i8"
  TEST "concatenate_rank4_i8.mlir"
  INSTANCES
    "(2,2,3,2)(2,2,3,4)"
    "(2,3,4,50)(2,3,4,10)"
    "(1,1,5,400)(1,1,5,100)"
)

op_tests(
  NAME "convolution_rank2_i8"
  TEST "convolution_rank2_i8.mlir"
  INSTANCES
    "(4,8)(8,4)"
    "(120,256)(256,300), [manual]"
    "(300,100)(100,450), [manual]"
)

op_tests(
  NAME "convolution_rank3_i8"
  TEST "convolution_rank3_i8.mlir"
  INSTANCES
    "(1,8,3)(3,3,16)"
    "(2,120,3)(5,3,32)"
    "(1,300,3)(3,3,8)"
)

op_tests(
  NAME "convolution_rank4_i8"
  TEST "convolution_rank4_i8.mlir"
  INSTANCES
    "(1,8,8,3)(3,3,3,16)"
    "(2,12,12,3)(5,5,3,32), [manual]"
    "(1,50,50,3)(3,3,3,8)"
)

op_tests(
  NAME "divide_rank1_i8"
  TEST "divide_rank1_i8.mlir"
  INSTANCES
    "(8)(8)"
    "(256)(256)"
    "(450)(450)"
)

op_tests(
  NAME "divide_rank2_i8"
  TEST "divide_rank2_i8.mlir"
  INSTANCES
    "(4,8)(4,8)"
    "(120,256)(120,256)"
    "(300,450)(300,450)"
)

op_tests(
  NAME "divide_rank3_i8"
  TEST "divide_rank3_i8.mlir"
  INSTANCES
    "(2,3,4)(2,3,4)"
    "(10,20,30)(10,20,30)"
    "(5,100,2)(5,100,2)"
)

op_tests(
  NAME "divide_rank4_i8"
  TEST "divide_rank4_i8.mlir"
  INSTANCES
    "(2,2,3,2)(2,2,3,2)"
    "(2,3,4,50)(2,3,4,50)"
    "(1,1,5,400)(1,1,5,400)"
)

op_tests(
  NAME "dynamic_broadcast_in_dim_rank2_i8"
  TEST "dynamic_broadcast_in_dim_rank2_i8.mlir"
  INSTANCES
    "(8)"
    "(256)"
    "(450)"
)

op_tests(
  NAME "dynamic_broadcast_in_dim_rank3_i8"
  TEST "dynamic_broadcast_in_dim_rank3_i8.mlir"
  INSTANCES
    "(4,8)"
    "(120,256)"
    "(300,450), [manual]"
)

op_tests(
  NAME "dynamic_broadcast_in_dim_rank4_i8"
  TEST "dynamic_broadcast_in_dim_rank4_i8.mlir"
  INSTANCES
    "(2,3,4)"
    "(10,20,30)"
    "(5,100,2)"
)

op_tests(
  NAME "matmul_rank1_i8"
  TEST "matmul_rank1_i8.mlir"
  INSTANCES
    "(8)(8)"
    "(256)(256)"
)

op_tests(
  NAME "matmul_rank2_i8"
  TEST "matmul_rank2_i8.mlir"
  INSTANCES
    "(4,8)(8,4)"
    "(120,256)(256,300), [manual]"
    "(300,100)(100,450), [manual]"
)

op_tests(
  NAME "maximum_rank1_i8"
  TEST "maximum_rank1_i8.mlir"
  INSTANCES
    "(8)(8)"
    "(256)(256)"
    "(450)(450)"
)

op_tests(
  NAME "maximum_rank2_i8"
  TEST "maximum_rank2_i8.mlir"
  INSTANCES
    "(4,8)(4,8)"
    "(120,256)(120,256)"
    "(300,450)(300,450)"
)

op_tests(
  NAME "maximum_rank3_i8"
  TEST "maximum_rank3_i8.mlir"
  INSTANCES
    "(2,3,4)(2,3,4)"
    "(10,20,30)(10,20,30)"
    "(5,100,2)(5,100,2)"
)

op_tests(
  NAME "maximum_rank4_i8"
  TEST "maximum_rank4_i8.mlir"
  INSTANCES
    "(2,2,3,2)(2,2,3,2)"
    "(2,3,4,50)(2,3,4,50)"
    "(1,1,5,400)(1,1,5,400)"
)

op_tests(
  NAME "minimum_rank1_i8"
  TEST "minimum_rank1_i8.mlir"
  INSTANCES
    "(8)(8)"
    "(256)(256)"
    "(450)(450)"
)

op_tests(
  NAME "minimum_rank2_i8"
  TEST "minimum_rank2_i8.mlir"
  INSTANCES
    "(4,8)(4,8)"
    "(120,256)(120,256)"
    "(300,450)(300,450)"
)

op_tests(
  NAME "minimum_rank3_i8"
  TEST "minimum_rank3_i8.mlir"
  INSTANCES
    "(2,3,4)(2,3,4)"
    "(10,20,30)(10,20,30)"
    "(5,100,2)(5,100,2)"
)

op_tests(
  NAME "minimum_rank4_i8"
  TEST "minimum_rank4_i8.mlir"
  INSTANCES
    "(2,2,3,2)(2,2,3,2)"
    "(2,3,4,50)(2,3,4,50)"
    "(1,1,5,400)(1,1,5,400)"
)

op_tests(
  NAME "multiply_rank1_i8"
  TEST "multiply_rank1_i8.mlir"
  INSTANCES
    "(8)(8)"
    "(256)(256)"
    "(450)(450)"
)

op_tests(
  NAME "multiply_rank2_i8"
  TEST "multiply_rank2_i8.mlir"
  INSTANCES
    "(4,8)(4,8)"
    "(120,256)(120,256)"
    "(300,450)(300,450)"
)

op_tests(
  NAME "multiply_rank3_i8"
  TEST "multiply_rank3_i8.mlir"
  INSTANCES
    "(2,3,4)(2,3,4)"
    "(10,20,30)(10,20,30)"
    "(5,100,2)(5,100,2)"
)

op_tests(
  NAME "multiply_rank4_i8"
  TEST "multiply_rank4_i8.mlir"
  INSTANCES
    "(2,2,3,2)(2,2,3,2)"
    "(2,3,4,50)(2,3,4,50)"
    "(1,1,5,400)(1,1,5,400)"
)

op_tests(
  NAME "negate_rank1_i8"
  TEST "negate_rank1_i8.mlir"
  INSTANCES
    "(8)"
    "(256)"
    "(450)"
)

op_tests(
  NAME "negate_rank2_i8"
  TEST "negate_rank2_i8.mlir"
  INSTANCES
    "(4,8)"
    "(120,256)"
    "(300,450)"
)

op_tests(
  NAME "negate_rank3_i8"
  TEST "negate_rank3_i8.mlir"
  INSTANCES
    "(2,3,4)"
    "(10,20,30)"
    "(5,100,2)"
)

op_tests(
  NAME "negate_rank4_i8"
  TEST "negate_rank4_i8.mlir"
  INSTANCES
    "(2,2,3,2)"
    "(2,3,4,50)"
    "(1,1,5,400)"
)

op_tests(
  NAME "reduce_sum_rank1_i8"
  TEST "reduce_sum_rank1_i8.mlir"
  INSTANCES
    "(8)"
    "(256)"
    "(450)"
)

op_tests(
  NAME "reduce_sum_rank2_i8"
  TEST "reduce_sum_rank2_i8.mlir"
  INSTANCES
    "(4,8)"
    "(120,256)"
    "(300,450)"
)

op_tests(
  NAME "reduce_sum_rank3_i8"
  TEST "reduce_sum_rank3_i8.mlir"
  INSTANCES
    "(2,3,4)"
    "(10,20,30)"
    "(5,100,2)"
)

op_tests(
  NAME "reduce_sum_rank4_i8"
  TEST "reduce_sum_rank4_i8.mlir"
  INSTANCES
    "(2,2,3,2)"
    "(2,3,4,50)"
    "(1,1,5,400)"
)

op_tests(
  NAME "select_rank1_i8"
  TEST "select_rank1_i8.mlir"
  INSTANCES
    "(8)(8)(8)"
    "(256)(256)(256)"
    "(450)(450)(450)"
)

op_tests(
  NAME "select_rank2_i8"
  TEST "select_rank2_i8.mlir"
  INSTANCES
    "(4,8)(4,8)(4,8)"
    "(120,256)(120,256)(120,256)"
    "(300,450)(300,450)(300,450)"
)

op_tests(
  NAME "select_rank3_i8"
  TEST "select_rank3_i8.mlir"
  INSTANCES
    "(2,3,4)(2,3,4)(2,3,4)"
    "(10,20,30)(10,20,30)(10,20,30)"
    "(5,100,2)(5,100,2)(5,100,2)"
)

op_tests(
  NAME "select_rank4_i8"
  TEST "select_rank4_i8.mlir"
  INSTANCES
    "(2,2,3,2)(2,2,3,2)(2,2,3,2)"
    "(2,3,4,50)(2,3,4,50)(2,3,4,50)"
    "(1,1,5,400)(1,1,5,400)(1,1,5,400)"
)

op_tests(
  NAME "slice_rank1_i8"
  TEST "slice_rank1_i8.mlir"
  INSTANCES
    "(8)"
    "(256)"
    "(450)"
)

op_tests(
  NAME "slice_rank2_i8"
  TEST "slice_rank2_i8.mlir"
  INSTANCES
    "(4,8)"
    "(120,256)"
    "(300,450)"
)

op_tests(
  NAME "slice_rank3_i8"
  TEST "slice_rank3_i8.mlir"
  INSTANCES
    "(2,3,4)"
    "(10,20,30)"
    "(5,100,4)"
)

op_tests(
  NAME "slice_rank4_i8"
  TEST "slice_rank4_i8.mlir"
  INSTANCES
    "(2,2,3,4)"
    "(2,3,4,50)"
    "(2,2,5,400)"
)

op_tests(
  NAME "subtract_rank1_i8"
  TEST "subtract_rank1_i8.mlir"
  INSTANCES
    "(8)(8)"
    "(256)(256)"
    "(450)(450)"
)

op_tests(
  NAME "subtract_rank2_i8"
  TEST "subtract_rank2_i8.mlir"
  INSTANCES
    "(4,8)(4,8)"
    "(120,256)(120,256)"
    "(300,450)(300,450)"
)

op_tests(
  NAME "subtract_rank3_i8"
  TEST "subtract_rank3_i8.mlir"
  INSTANCES
    "(2,3,4)(2,3,4)"
    "(10,20,30)(10,20,30)"
    "(5,100,2)(5,100,2)"
)

op_tests(
  NAME "subtract_rank4_i8"
  TEST "subtract_rank4_i8.mlir"
  INSTANCES
    "(2,2,3,2)(2,2,3,2)"
    "(2,3,4,50)(2,3,4,50)"
    "(1,1,5,400)(1,1,5,400)"
)

op_tests(
  NAME "transpose_rank2_i8"
  TEST "transpose_rank2_i8.mlir"
  INSTANCES
    "(4,8)"
    "(120,256)"
    "(300,450)"
)

op_tests(
  NAME "transpose_rank3_i8"
  TEST "transpose_rank3_i8.mlir"
  INSTANCES
    "(2,3,4)"
    "(10,20,30)"
    "(5,100,2)"
)

op_tests(
  NAME "transpose_rank4_i8"
  TEST "transpose_rank4_i8.mlir"
  INSTANCES
    "(2,2,3,2)"
    "(2,3,4,50)"
    "(1,1,5,400)"
)

