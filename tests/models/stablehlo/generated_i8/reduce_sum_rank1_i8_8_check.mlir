module {
  func.func @reduce_sum_rank1_i8_8() {
    %cst = arith.constant dense<28> : tensor<i8>
    %c = stablehlo.constant dense<0> : tensor<i8>
    %cst_0 = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi8>
    %0 = util.optimization_barrier %cst_0 : tensor<8xi8>
    %1 = stablehlo.reduce(%0 init: %c) applies stablehlo.add across dimensions = [0] : (tensor<8xi8>, tensor<i8>) -> tensor<i8>
    %2 = util.optimization_barrier %cst : tensor<i8>
    "check.expect_eq"(%1, %2) : (tensor<i8>, tensor<i8>) -> ()
    return
  }
}
