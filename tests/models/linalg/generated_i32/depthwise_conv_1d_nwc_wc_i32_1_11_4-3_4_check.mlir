module {
  func.func @"depthwise_conv_1d_nwc_wc_i32_1_11_4-3_4"() {
    %cst = arith.constant dense<[[[80, 107, 140, 179], [128, 167, 212, 263], [176, 227, 284, 347], [224, 287, 356, 431], [272, 347, 428, 515], [320, 407, 500, 599], [368, 467, 572, 683], [416, 527, 644, 767], [464, 587, 716, 851]]]> : tensor<1x9x4xi32>
    %cst_0 = arith.constant dense<[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]> : tensor<3x4xi32>
    %c0_i32 = arith.constant 0 : i32
    %cst_1 = arith.constant dense<[[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23], [24, 25, 26, 27], [28, 29, 30, 31], [32, 33, 34, 35], [36, 37, 38, 39], [40, 41, 42, 43]]]> : tensor<1x11x4xi32>
    %0 = util.optimization_barrier %cst_1 : tensor<1x11x4xi32>
    %1 = util.optimization_barrier %cst_0 : tensor<3x4xi32>
    %2 = tensor.empty() : tensor<1x9x4xi32>
    %3 = linalg.fill ins(%c0_i32 : i32) outs(%2 : tensor<1x9x4xi32>) -> tensor<1x9x4xi32>
    %4 = linalg.depthwise_conv_1d_nwc_wc {dilations = dense<1> : vector<1xi64>, strides = dense<1> : vector<1xi64>} ins(%0, %1 : tensor<1x11x4xi32>, tensor<3x4xi32>) outs(%3 : tensor<1x9x4xi32>) -> tensor<1x9x4xi32>
    %5 = util.optimization_barrier %cst : tensor<1x9x4xi32>
    "check.expect_eq"(%4, %5) : (tensor<1x9x4xi32>, tensor<1x9x4xi32>) -> ()
    return
  }
}
