module {
  func.func @not_rank2_i32_4_8() {
    %cst = arith.constant dense<[[-1, -2, -3, -4, -5, -6, -7, -8], [-9, -10, -11, -12, -13, -14, -15, -16], [-17, -18, -19, -20, -21, -22, -23, -24], [-25, -26, -27, -28, -29, -30, -31, -32]]> : tensor<4x8xi32>
    %cst_0 = arith.constant dense<[[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15], [16, 17, 18, 19, 20, 21, 22, 23], [24, 25, 26, 27, 28, 29, 30, 31]]> : tensor<4x8xi32>
    %0 = util.optimization_barrier %cst_0 : tensor<4x8xi32>
    %1 = stablehlo.not %0 : tensor<4x8xi32>
    %2 = util.optimization_barrier %cst : tensor<4x8xi32>
    "check.expect_eq"(%1, %2) : (tensor<4x8xi32>, tensor<4x8xi32>) -> ()
    return
  }
}
