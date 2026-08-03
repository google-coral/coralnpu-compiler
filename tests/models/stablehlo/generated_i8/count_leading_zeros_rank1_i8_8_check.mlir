module {
  func.func @count_leading_zeros_rank1_i8_8() {
    %cst = arith.constant dense<[8, 7, 6, 6, 5, 5, 5, 5]> : tensor<8xi8>
    %cst_0 = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi8>
    %0 = util.optimization_barrier %cst_0 : tensor<8xi8>
    %1 = stablehlo.count_leading_zeros %0 : tensor<8xi8>
    %2 = util.optimization_barrier %cst : tensor<8xi8>
    "check.expect_eq"(%1, %2) : (tensor<8xi8>, tensor<8xi8>) -> ()
    return
  }
}
