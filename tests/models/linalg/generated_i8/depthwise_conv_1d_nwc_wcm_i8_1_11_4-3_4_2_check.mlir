module {
  func.func @"depthwise_conv_1d_nwc_wcm_i8_1_11_4-3_4_2"() {
    %cst = arith.constant dense<[[[[-96, -84], [-42, -27], [24, 42], [102, 123]], [[0, 24], [78, 105], [-88, -58], [14, 47]], [[96, -124], [-58, -19], [56, 98], [-74, -29]], [[-64, -16], [62, 113], [-56, -2], [94, -105]], [[32, 92], [-74, -11], [88, -102], [6, 75]], [[-128, -56], [46, 121], [-24, 54], [-82, -1]], [[-32, 52], [-90, -3], [120, -46], [86, -77]], [[64, -96], [30, -127], [8, 110], [-2, 103]], [[-96, 12], [-106, 5], [-104, 10], [-90, 27]]]]> : tensor<1x9x4x2xi8>
    %cst_0 = arith.constant dense<[[[0, 1], [2, 3], [4, 5], [6, 7]], [[8, 9], [10, 11], [12, 13], [14, 15]], [[16, 17], [18, 19], [20, 21], [22, 23]]]> : tensor<3x4x2xi8>
    %c0_i8 = arith.constant 0 : i8
    %cst_1 = arith.constant dense<[[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23], [24, 25, 26, 27], [28, 29, 30, 31], [32, 33, 34, 35], [36, 37, 38, 39], [40, 41, 42, 43]]]> : tensor<1x11x4xi8>
    %0 = util.optimization_barrier %cst_1 : tensor<1x11x4xi8>
    %1 = util.optimization_barrier %cst_0 : tensor<3x4x2xi8>
    %2 = tensor.empty() : tensor<1x9x4x2xi8>
    %3 = linalg.fill ins(%c0_i8 : i8) outs(%2 : tensor<1x9x4x2xi8>) -> tensor<1x9x4x2xi8>
    %4 = linalg.depthwise_conv_1d_nwc_wcm {dilations = dense<1> : vector<1xi64>, strides = dense<1> : vector<1xi64>} ins(%0, %1 : tensor<1x11x4xi8>, tensor<3x4x2xi8>) outs(%3 : tensor<1x9x4x2xi8>) -> tensor<1x9x4x2xi8>
    %5 = util.optimization_barrier %cst : tensor<1x9x4x2xi8>
    "check.expect_eq"(%4, %5) : (tensor<1x9x4x2xi8>, tensor<1x9x4x2xi8>) -> ()
    return
  }
}
