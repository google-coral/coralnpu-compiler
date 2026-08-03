module {
  func.func @reverse_rank2_i32_4_8() {
    %cst = arith.constant dense<[[7, 6, 5, 4, 3, 2, 1, 0], [15, 14, 13, 12, 11, 10, 9, 8], [23, 22, 21, 20, 19, 18, 17, 16], [31, 30, 29, 28, 27, 26, 25, 24]]> : tensor<4x8xi32>
    %cst_0 = arith.constant dense<[[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15], [16, 17, 18, 19, 20, 21, 22, 23], [24, 25, 26, 27, 28, 29, 30, 31]]> : tensor<4x8xi32>
    %0 = util.optimization_barrier %cst_0 : tensor<4x8xi32>
    %1 = stablehlo.reverse %0, dims = [1] : tensor<4x8xi32>
    %2 = util.optimization_barrier %cst : tensor<4x8xi32>
    "check.expect_eq"(%1, %2) : (tensor<4x8xi32>, tensor<4x8xi32>) -> ()
    return
  }
}
