module {
  func.func @"divide_rank3_i16_2_3_4-2_3_4"() {
    %cst = arith.constant dense<[[[-1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]]> : tensor<2x3x4xi16>
    %cst_0 = arith.constant dense<[[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]> : tensor<2x3x4xi16>
    %0 = util.optimization_barrier %cst_0 : tensor<2x3x4xi16>
    %1 = util.optimization_barrier %cst_0 : tensor<2x3x4xi16>
    %2 = stablehlo.divide %0, %1 : tensor<2x3x4xi16>
    %3 = util.optimization_barrier %cst : tensor<2x3x4xi16>
    "check.expect_eq"(%2, %3) : (tensor<2x3x4xi16>, tensor<2x3x4xi16>) -> ()
    return
  }
}
