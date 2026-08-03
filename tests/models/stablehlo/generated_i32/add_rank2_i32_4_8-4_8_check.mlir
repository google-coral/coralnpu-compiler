module {
  func.func @"add_rank2_i32_4_8-4_8"() {
    %cst = arith.constant dense<[[0, 2, 4, 6, 8, 10, 12, 14], [16, 18, 20, 22, 24, 26, 28, 30], [32, 34, 36, 38, 40, 42, 44, 46], [48, 50, 52, 54, 56, 58, 60, 62]]> : tensor<4x8xi32>
    %cst_0 = arith.constant dense<[[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15], [16, 17, 18, 19, 20, 21, 22, 23], [24, 25, 26, 27, 28, 29, 30, 31]]> : tensor<4x8xi32>
    %0 = util.optimization_barrier %cst_0 : tensor<4x8xi32>
    %1 = util.optimization_barrier %cst_0 : tensor<4x8xi32>
    %2 = stablehlo.add %0, %1 : tensor<4x8xi32>
    %3 = util.optimization_barrier %cst : tensor<4x8xi32>
    "check.expect_eq"(%2, %3) : (tensor<4x8xi32>, tensor<4x8xi32>) -> ()
    return
  }
}
