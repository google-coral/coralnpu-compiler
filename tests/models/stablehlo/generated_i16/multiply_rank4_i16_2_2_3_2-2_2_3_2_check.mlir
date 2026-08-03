module {
  func.func @"multiply_rank4_i16_2_2_3_2-2_2_3_2"() {
    %cst = arith.constant dense<[[[[0, 1], [4, 9], [16, 25]], [[36, 49], [64, 81], [100, 121]]], [[[144, 169], [196, 225], [256, 289]], [[324, 361], [400, 441], [484, 529]]]]> : tensor<2x2x3x2xi16>
    %cst_0 = arith.constant dense<[[[[0, 1], [2, 3], [4, 5]], [[6, 7], [8, 9], [10, 11]]], [[[12, 13], [14, 15], [16, 17]], [[18, 19], [20, 21], [22, 23]]]]> : tensor<2x2x3x2xi16>
    %0 = util.optimization_barrier %cst_0 : tensor<2x2x3x2xi16>
    %1 = util.optimization_barrier %cst_0 : tensor<2x2x3x2xi16>
    %2 = stablehlo.multiply %0, %1 : tensor<2x2x3x2xi16>
    %3 = util.optimization_barrier %cst : tensor<2x2x3x2xi16>
    "check.expect_eq"(%2, %3) : (tensor<2x2x3x2xi16>, tensor<2x2x3x2xi16>) -> ()
    return
  }
}
