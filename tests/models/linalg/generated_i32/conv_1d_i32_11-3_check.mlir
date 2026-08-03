module {
  func.func @"conv_1d_i32_11-3"() {
    %cst = arith.constant dense<[5, 8, 11, 14, 17, 20, 23, 26, 29]> : tensor<9xi32>
    %cst_0 = arith.constant dense<[0, 1, 2]> : tensor<3xi32>
    %c0_i32 = arith.constant 0 : i32
    %cst_1 = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]> : tensor<11xi32>
    %0 = util.optimization_barrier %cst_1 : tensor<11xi32>
    %1 = util.optimization_barrier %cst_0 : tensor<3xi32>
    %2 = tensor.empty() : tensor<9xi32>
    %3 = linalg.fill ins(%c0_i32 : i32) outs(%2 : tensor<9xi32>) -> tensor<9xi32>
    %4 = linalg.conv_1d ins(%0, %1 : tensor<11xi32>, tensor<3xi32>) outs(%3 : tensor<9xi32>) -> tensor<9xi32>
    %5 = util.optimization_barrier %cst : tensor<9xi32>
    "check.expect_eq"(%4, %5) : (tensor<9xi32>, tensor<9xi32>) -> ()
    return
  }
}
