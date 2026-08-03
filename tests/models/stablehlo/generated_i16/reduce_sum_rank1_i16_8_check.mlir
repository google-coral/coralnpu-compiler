module {
  func.func @reduce_sum_rank1_i16_8() {
    %cst = arith.constant dense<28> : tensor<i16>
    %c = stablehlo.constant dense<0> : tensor<i16>
    %cst_0 = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi16>
    %0 = util.optimization_barrier %cst_0 : tensor<8xi16>
    %1 = stablehlo.reduce(%0 init: %c) applies stablehlo.add across dimensions = [0] : (tensor<8xi16>, tensor<i16>) -> tensor<i16>
    %2 = util.optimization_barrier %cst : tensor<i16>
    "check.expect_eq"(%1, %2) : (tensor<i16>, tensor<i16>) -> ()
    return
  }
}
