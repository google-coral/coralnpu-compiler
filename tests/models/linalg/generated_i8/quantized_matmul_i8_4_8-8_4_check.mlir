module {
  func.func @"quantized_matmul_i8_4_8-8_4"() {
    %cst = arith.constant dense<[[372, 384, 396, 408], [1460, 1536, 1612, 1688], [2548, 2688, 2828, 2968], [3636, 3840, 4044, 4248]]> : tensor<4x4xi32>
    %cst_0 = arith.constant dense<[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23], [24, 25, 26, 27], [28, 29, 30, 31]]> : tensor<8x4xi8>
    %c-3_i32 = arith.constant -3 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_1 = arith.constant dense<[[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15], [16, 17, 18, 19, 20, 21, 22, 23], [24, 25, 26, 27, 28, 29, 30, 31]]> : tensor<4x8xi8>
    %0 = util.optimization_barrier %cst_1 : tensor<4x8xi8>
    %1 = util.optimization_barrier %cst_0 : tensor<8x4xi8>
    %2 = tensor.empty() : tensor<4x4xi32>
    %3 = linalg.fill ins(%c0_i32 : i32) outs(%2 : tensor<4x4xi32>) -> tensor<4x4xi32>
    %4 = linalg.quantized_matmul ins(%0, %1, %c2_i32, %c-3_i32 : tensor<4x8xi8>, tensor<8x4xi8>, i32, i32) outs(%3 : tensor<4x4xi32>) -> tensor<4x4xi32>
    %5 = util.optimization_barrier %cst : tensor<4x4xi32>
    "check.expect_eq"(%4, %5) : (tensor<4x4xi32>, tensor<4x4xi32>) -> ()
    return
  }
}
