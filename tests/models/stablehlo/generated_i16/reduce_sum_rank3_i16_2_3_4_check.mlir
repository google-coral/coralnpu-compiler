module {
  func.func @reduce_sum_rank3_i16_2_3_4() {
    %cst = arith.constant dense<[[6, 22, 38], [54, 70, 86]]> : tensor<2x3xi16>
    %c = stablehlo.constant dense<0> : tensor<i16>
    %cst_0 = arith.constant dense<[[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]> : tensor<2x3x4xi16>
    %0 = util.optimization_barrier %cst_0 : tensor<2x3x4xi16>
    %1 = stablehlo.reduce(%0 init: %c) applies stablehlo.add across dimensions = [2] : (tensor<2x3x4xi16>, tensor<i16>) -> tensor<2x3xi16>
    %2 = util.optimization_barrier %cst : tensor<2x3xi16>
    "check.expect_eq"(%1, %2) : (tensor<2x3xi16>, tensor<2x3xi16>) -> ()
    return
  }
}
