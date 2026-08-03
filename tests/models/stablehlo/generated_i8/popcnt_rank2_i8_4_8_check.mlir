module {
  func.func @popcnt_rank2_i8_4_8() {
    %cst = arith.constant dense<[[0, 1, 1, 2, 1, 2, 2, 3], [1, 2, 2, 3, 2, 3, 3, 4], [1, 2, 2, 3, 2, 3, 3, 4], [2, 3, 3, 4, 3, 4, 4, 5]]> : tensor<4x8xi8>
    %cst_0 = arith.constant dense<[[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15], [16, 17, 18, 19, 20, 21, 22, 23], [24, 25, 26, 27, 28, 29, 30, 31]]> : tensor<4x8xi8>
    %0 = util.optimization_barrier %cst_0 : tensor<4x8xi8>
    %1 = stablehlo.popcnt %0 : tensor<4x8xi8>
    %2 = util.optimization_barrier %cst : tensor<4x8xi8>
    "check.expect_eq"(%1, %2) : (tensor<4x8xi8>, tensor<4x8xi8>) -> ()
    return
  }
}
