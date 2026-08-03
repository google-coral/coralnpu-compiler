module {
  func.func @"minimum_rank4_i32_2_2_3_2-2_2_3_2"() {
    %cst = arith.constant dense<[[[[0, 1], [2, 3], [4, 5]], [[6, 7], [8, 9], [10, 11]]], [[[12, 13], [14, 15], [16, 17]], [[18, 19], [20, 21], [22, 23]]]]> : tensor<2x2x3x2xi32>
    %0 = util.optimization_barrier %cst : tensor<2x2x3x2xi32>
    %1 = util.optimization_barrier %cst : tensor<2x2x3x2xi32>
    %2 = stablehlo.minimum %0, %1 : tensor<2x2x3x2xi32>
    %3 = util.optimization_barrier %cst : tensor<2x2x3x2xi32>
    "check.expect_eq"(%2, %3) : (tensor<2x2x3x2xi32>, tensor<2x2x3x2xi32>) -> ()
    return
  }
}
