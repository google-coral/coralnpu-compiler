module {
  func.func @"maximum_rank4_i8_2_2_3_2-2_2_3_2"() {
    %cst = arith.constant dense<[[[[0, 1], [2, 3], [4, 5]], [[6, 7], [8, 9], [10, 11]]], [[[12, 13], [14, 15], [16, 17]], [[18, 19], [20, 21], [22, 23]]]]> : tensor<2x2x3x2xi8>
    %0 = util.optimization_barrier %cst : tensor<2x2x3x2xi8>
    %1 = util.optimization_barrier %cst : tensor<2x2x3x2xi8>
    %2 = stablehlo.maximum %0, %1 : tensor<2x2x3x2xi8>
    %3 = util.optimization_barrier %cst : tensor<2x2x3x2xi8>
    "check.expect_eq"(%2, %3) : (tensor<2x2x3x2xi8>, tensor<2x2x3x2xi8>) -> ()
    return
  }
}
