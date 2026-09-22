module {
  func.func @"matmul_i8_4_8-8_4"() {
    %cst = arith.constant dense<[[48, 76, 104, -124], [-80, 12, 104, -60], [48, -52, 104, 4], [-80, -116, 104, 68]]> : tensor<4x4xi8>
    %cst_0 = arith.constant dense<[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23], [24, 25, 26, 27], [28, 29, 30, 31]]> : tensor<8x4xi8>
    %c0_i8 = arith.constant 0 : i8
    %cst_1 = arith.constant dense<[[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15], [16, 17, 18, 19, 20, 21, 22, 23], [24, 25, 26, 27, 28, 29, 30, 31]]> : tensor<4x8xi8>
    %0 = util.optimization_barrier %cst_1 : tensor<4x8xi8>
    %1 = util.optimization_barrier %cst_0 : tensor<8x4xi8>
    %2 = tensor.empty() : tensor<4x4xi8>
    %3 = linalg.fill ins(%c0_i8 : i8) outs(%2 : tensor<4x4xi8>) -> tensor<4x4xi8>
    %4 = linalg.matmul ins(%0, %1 : tensor<4x8xi8>, tensor<8x4xi8>) outs(%3 : tensor<4x4xi8>) -> tensor<4x4xi8>
    %5 = util.optimization_barrier %cst : tensor<4x4xi8>
    "check.expect_eq"(%4, %5) : (tensor<4x4xi8>, tensor<4x4xi8>) -> ()
    return
  }
}
