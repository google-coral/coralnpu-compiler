module {
  func.func @"add_rank4_i16_2_2_3_2-2_2_3_2"() {
    %cst = arith.constant dense<[[[[0, 2], [4, 6], [8, 10]], [[12, 14], [16, 18], [20, 22]]], [[[24, 26], [28, 30], [32, 34]], [[36, 38], [40, 42], [44, 46]]]]> : tensor<2x2x3x2xi16>
    %cst_0 = arith.constant dense<[[[[0, 1], [2, 3], [4, 5]], [[6, 7], [8, 9], [10, 11]]], [[[12, 13], [14, 15], [16, 17]], [[18, 19], [20, 21], [22, 23]]]]> : tensor<2x2x3x2xi16>
    %0 = util.optimization_barrier %cst_0 : tensor<2x2x3x2xi16>
    %1 = util.optimization_barrier %cst_0 : tensor<2x2x3x2xi16>
    %2 = stablehlo.add %0, %1 : tensor<2x2x3x2xi16>
    %3 = util.optimization_barrier %cst : tensor<2x2x3x2xi16>
    "check.expect_eq"(%2, %3) : (tensor<2x2x3x2xi16>, tensor<2x2x3x2xi16>) -> ()
    return
  }
}
