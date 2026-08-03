module {
  func.func @count_leading_zeros_rank2_i8_4_8() {
    %cst = arith.constant dense<[[8, 7, 6, 6, 5, 5, 5, 5], [4, 4, 4, 4, 4, 4, 4, 4], [3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3]]> : tensor<4x8xi8>
    %cst_0 = arith.constant dense<[[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15], [16, 17, 18, 19, 20, 21, 22, 23], [24, 25, 26, 27, 28, 29, 30, 31]]> : tensor<4x8xi8>
    %0 = util.optimization_barrier %cst_0 : tensor<4x8xi8>
    %1 = stablehlo.count_leading_zeros %0 : tensor<4x8xi8>
    %2 = util.optimization_barrier %cst : tensor<4x8xi8>
    "check.expect_eq"(%1, %2) : (tensor<4x8xi8>, tensor<4x8xi8>) -> ()
    return
  }
}
