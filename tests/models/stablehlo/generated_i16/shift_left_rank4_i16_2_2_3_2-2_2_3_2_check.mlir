module {
  func.func @"shift_left_rank4_i16_2_2_3_2-2_2_3_2"() {
    %cst = arith.constant dense<[[[[0, 2], [8, 24], [64, 160]], [[384, 896], [2048, 4608], [10240, 22528]]], [[[-16384, -24576], [-32768, -32768], [0, 0]], [[0, 0], [0, 0], [0, 0]]]]> : tensor<2x2x3x2xi16>
    %cst_0 = arith.constant dense<[[[[0, 1], [2, 3], [4, 5]], [[6, 7], [8, 9], [10, 11]]], [[[12, 13], [14, 15], [16, 17]], [[18, 19], [20, 21], [22, 23]]]]> : tensor<2x2x3x2xi16>
    %0 = util.optimization_barrier %cst_0 : tensor<2x2x3x2xi16>
    %1 = util.optimization_barrier %cst_0 : tensor<2x2x3x2xi16>
    %2 = stablehlo.shift_left %0, %1 : tensor<2x2x3x2xi16>
    %3 = util.optimization_barrier %cst : tensor<2x2x3x2xi16>
    "check.expect_eq"(%2, %3) : (tensor<2x2x3x2xi16>, tensor<2x2x3x2xi16>) -> ()
    return
  }
}
