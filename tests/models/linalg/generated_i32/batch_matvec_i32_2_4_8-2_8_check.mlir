module {
  func.func @"batch_matvec_i32_2_4_8-2_8"() {
    %cst = arith.constant dense<[[140, 364, 588, 812], [3308, 4044, 4780, 5516]]> : tensor<2x4xi32>
    %cst_0 = arith.constant dense<[[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]]> : tensor<2x8xi32>
    %c0_i32 = arith.constant 0 : i32
    %cst_1 = arith.constant dense<[[[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15], [16, 17, 18, 19, 20, 21, 22, 23], [24, 25, 26, 27, 28, 29, 30, 31]], [[32, 33, 34, 35, 36, 37, 38, 39], [40, 41, 42, 43, 44, 45, 46, 47], [48, 49, 50, 51, 52, 53, 54, 55], [56, 57, 58, 59, 60, 61, 62, 63]]]> : tensor<2x4x8xi32>
    %0 = util.optimization_barrier %cst_1 : tensor<2x4x8xi32>
    %1 = util.optimization_barrier %cst_0 : tensor<2x8xi32>
    %2 = tensor.empty() : tensor<2x4xi32>
    %3 = linalg.fill ins(%c0_i32 : i32) outs(%2 : tensor<2x4xi32>) -> tensor<2x4xi32>
    %4 = linalg.batch_matvec ins(%0, %1 : tensor<2x4x8xi32>, tensor<2x8xi32>) outs(%3 : tensor<2x4xi32>) -> tensor<2x4xi32>
    %5 = util.optimization_barrier %cst : tensor<2x4xi32>
    "check.expect_eq"(%4, %5) : (tensor<2x4xi32>, tensor<2x4xi32>) -> ()
    return
  }
}
