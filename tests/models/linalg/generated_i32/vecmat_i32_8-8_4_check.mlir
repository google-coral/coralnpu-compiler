module {
  func.func @"vecmat_i32_8-8_4"() {
    %cst = arith.constant dense<[560, 588, 616, 644]> : tensor<4xi32>
    %cst_0 = arith.constant dense<[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23], [24, 25, 26, 27], [28, 29, 30, 31]]> : tensor<8x4xi32>
    %c0_i32 = arith.constant 0 : i32
    %cst_1 = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi32>
    %0 = util.optimization_barrier %cst_1 : tensor<8xi32>
    %1 = util.optimization_barrier %cst_0 : tensor<8x4xi32>
    %2 = tensor.empty() : tensor<4xi32>
    %3 = linalg.fill ins(%c0_i32 : i32) outs(%2 : tensor<4xi32>) -> tensor<4xi32>
    %4 = linalg.vecmat ins(%0, %1 : tensor<8xi32>, tensor<8x4xi32>) outs(%3 : tensor<4xi32>) -> tensor<4xi32>
    %5 = util.optimization_barrier %cst : tensor<4xi32>
    "check.expect_eq"(%4, %5) : (tensor<4xi32>, tensor<4xi32>) -> ()
    return
  }
}
