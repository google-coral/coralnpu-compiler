module {
  func.func @"depthwise_conv_1d_nwc_wc_i8_1_11_4-3_4"() {
    %cst = arith.constant dense<[[[80, 107, -116, -77], [-128, -89, -44, 7], [-80, -29, 28, 91], [-32, 31, 100, -81], [16, 91, -84, 3], [64, -105, -12, 87], [112, -45, 60, -85], [-96, 15, -124, -1], [-48, 75, -52, 83]]]> : tensor<1x9x4xi8>
    %cst_0 = arith.constant dense<[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]> : tensor<3x4xi8>
    %c0_i8 = arith.constant 0 : i8
    %cst_1 = arith.constant dense<[[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23], [24, 25, 26, 27], [28, 29, 30, 31], [32, 33, 34, 35], [36, 37, 38, 39], [40, 41, 42, 43]]]> : tensor<1x11x4xi8>
    %0 = util.optimization_barrier %cst_1 : tensor<1x11x4xi8>
    %1 = util.optimization_barrier %cst_0 : tensor<3x4xi8>
    %2 = tensor.empty() : tensor<1x9x4xi8>
    %3 = linalg.fill ins(%c0_i8 : i8) outs(%2 : tensor<1x9x4xi8>) -> tensor<1x9x4xi8>
    %4 = linalg.depthwise_conv_1d_nwc_wc {dilations = dense<1> : vector<1xi64>, strides = dense<1> : vector<1xi64>} ins(%0, %1 : tensor<1x11x4xi8>, tensor<3x4xi8>) outs(%3 : tensor<1x9x4xi8>) -> tensor<1x9x4xi8>
    %5 = util.optimization_barrier %cst : tensor<1x9x4xi8>
    "check.expect_eq"(%4, %5) : (tensor<1x9x4xi8>, tensor<1x9x4xi8>) -> ()
    return
  }
}
