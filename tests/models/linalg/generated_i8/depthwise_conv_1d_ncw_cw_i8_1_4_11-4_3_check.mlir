module {
  func.func @"depthwise_conv_1d_ncw_cw_i8_1_4_11-4_3"() {
    %cst = arith.constant dense<[[[5, 8, 11, 14, 17, 20, 23, 26, 29], [-110, -98, -86, -74, -62, -50, -38, -26, -14], [-27, -6, 15, 36, 57, 78, 99, 120, -115], [-2, 28, 58, 88, 118, -108, -78, -48, -18]]]> : tensor<1x4x9xi8>
    %cst_0 = arith.constant dense<[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]> : tensor<4x3xi8>
    %c0_i8 = arith.constant 0 : i8
    %cst_1 = arith.constant dense<[[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21], [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32], [33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43]]]> : tensor<1x4x11xi8>
    %0 = util.optimization_barrier %cst_1 : tensor<1x4x11xi8>
    %1 = util.optimization_barrier %cst_0 : tensor<4x3xi8>
    %2 = tensor.empty() : tensor<1x4x9xi8>
    %3 = linalg.fill ins(%c0_i8 : i8) outs(%2 : tensor<1x4x9xi8>) -> tensor<1x4x9xi8>
    %4 = linalg.depthwise_conv_1d_ncw_cw {dilations = dense<1> : vector<1xi64>, strides = dense<1> : vector<1xi64>} ins(%0, %1 : tensor<1x4x11xi8>, tensor<4x3xi8>) outs(%3 : tensor<1x4x9xi8>) -> tensor<1x4x9xi8>
    %5 = util.optimization_barrier %cst : tensor<1x4x9xi8>
    "check.expect_eq"(%4, %5) : (tensor<1x4x9xi8>, tensor<1x4x9xi8>) -> ()
    return
  }
}
