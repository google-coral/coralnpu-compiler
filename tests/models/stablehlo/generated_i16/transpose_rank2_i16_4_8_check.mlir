module {
  func.func @transpose_rank2_i16_4_8() {
    %cst = arith.constant dense<[[0, 8, 16, 24], [1, 9, 17, 25], [2, 10, 18, 26], [3, 11, 19, 27], [4, 12, 20, 28], [5, 13, 21, 29], [6, 14, 22, 30], [7, 15, 23, 31]]> : tensor<8x4xi16>
    %cst_0 = arith.constant dense<[[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15], [16, 17, 18, 19, 20, 21, 22, 23], [24, 25, 26, 27, 28, 29, 30, 31]]> : tensor<4x8xi16>
    %0 = util.optimization_barrier %cst_0 : tensor<4x8xi16>
    %1 = stablehlo.transpose %0, dims = [1, 0] : (tensor<4x8xi16>) -> tensor<8x4xi16>
    %2 = util.optimization_barrier %cst : tensor<8x4xi16>
    "check.expect_eq"(%1, %2) : (tensor<8x4xi16>, tensor<8x4xi16>) -> ()
    return
  }
}
