module {
  func.func @"depthwise_conv_1d_ncw_cw_i32_1_4_11-4_3"() {
    %cst = arith.constant dense<[[[5, 8, 11, 14, 17, 20, 23, 26, 29], [146, 158, 170, 182, 194, 206, 218, 230, 242], [485, 506, 527, 548, 569, 590, 611, 632, 653], [1022, 1052, 1082, 1112, 1142, 1172, 1202, 1232, 1262]]]> : tensor<1x4x9xi32>
    %cst_0 = arith.constant dense<[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]> : tensor<4x3xi32>
    %c0_i32 = arith.constant 0 : i32
    %cst_1 = arith.constant dense<[[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21], [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32], [33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43]]]> : tensor<1x4x11xi32>
    %0 = util.optimization_barrier %cst_1 : tensor<1x4x11xi32>
    %1 = util.optimization_barrier %cst_0 : tensor<4x3xi32>
    %2 = tensor.empty() : tensor<1x4x9xi32>
    %3 = linalg.fill ins(%c0_i32 : i32) outs(%2 : tensor<1x4x9xi32>) -> tensor<1x4x9xi32>
    %4 = linalg.depthwise_conv_1d_ncw_cw {dilations = dense<1> : vector<1xi64>, strides = dense<1> : vector<1xi64>} ins(%0, %1 : tensor<1x4x11xi32>, tensor<4x3xi32>) outs(%3 : tensor<1x4x9xi32>) -> tensor<1x4x9xi32>
    %5 = util.optimization_barrier %cst : tensor<1x4x9xi32>
    "check.expect_eq"(%4, %5) : (tensor<1x4x9xi32>, tensor<1x4x9xi32>) -> ()
    return
  }
}
