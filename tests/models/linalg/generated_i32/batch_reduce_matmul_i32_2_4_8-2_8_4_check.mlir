module {
  func.func @"batch_reduce_matmul_i32_2_4_8-2_8_4"() {
    %cst = arith.constant dense<[[13792, 14104, 14416, 14728], [17632, 18072, 18512, 18952], [21472, 22040, 22608, 23176], [25312, 26008, 26704, 27400]]> : tensor<4x4xi32>
    %cst_0 = arith.constant dense<[[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23], [24, 25, 26, 27], [28, 29, 30, 31]], [[32, 33, 34, 35], [36, 37, 38, 39], [40, 41, 42, 43], [44, 45, 46, 47], [48, 49, 50, 51], [52, 53, 54, 55], [56, 57, 58, 59], [60, 61, 62, 63]]]> : tensor<2x8x4xi32>
    %c0_i32 = arith.constant 0 : i32
    %cst_1 = arith.constant dense<[[[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15], [16, 17, 18, 19, 20, 21, 22, 23], [24, 25, 26, 27, 28, 29, 30, 31]], [[32, 33, 34, 35, 36, 37, 38, 39], [40, 41, 42, 43, 44, 45, 46, 47], [48, 49, 50, 51, 52, 53, 54, 55], [56, 57, 58, 59, 60, 61, 62, 63]]]> : tensor<2x4x8xi32>
    %0 = util.optimization_barrier %cst_1 : tensor<2x4x8xi32>
    %1 = util.optimization_barrier %cst_0 : tensor<2x8x4xi32>
    %2 = tensor.empty() : tensor<4x4xi32>
    %3 = linalg.fill ins(%c0_i32 : i32) outs(%2 : tensor<4x4xi32>) -> tensor<4x4xi32>
    %4 = linalg.batch_reduce_matmul ins(%0, %1 : tensor<2x4x8xi32>, tensor<2x8x4xi32>) outs(%3 : tensor<4x4xi32>) -> tensor<4x4xi32>
    %5 = util.optimization_barrier %cst : tensor<4x4xi32>
    "check.expect_eq"(%4, %5) : (tensor<4x4xi32>, tensor<4x4xi32>) -> ()
    return
  }
}
