module {
  func.func @count_leading_zeros_rank4_i16_2_2_3_2() {
    %cst = arith.constant dense<[[[[16, 15], [14, 14], [13, 13]], [[13, 13], [12, 12], [12, 12]]], [[[12, 12], [12, 12], [11, 11]], [[11, 11], [11, 11], [11, 11]]]]> : tensor<2x2x3x2xi16>
    %cst_0 = arith.constant dense<[[[[0, 1], [2, 3], [4, 5]], [[6, 7], [8, 9], [10, 11]]], [[[12, 13], [14, 15], [16, 17]], [[18, 19], [20, 21], [22, 23]]]]> : tensor<2x2x3x2xi16>
    %0 = util.optimization_barrier %cst_0 : tensor<2x2x3x2xi16>
    %1 = stablehlo.count_leading_zeros %0 : tensor<2x2x3x2xi16>
    %2 = util.optimization_barrier %cst : tensor<2x2x3x2xi16>
    "check.expect_eq"(%1, %2) : (tensor<2x2x3x2xi16>, tensor<2x2x3x2xi16>) -> ()
    return
  }
}
