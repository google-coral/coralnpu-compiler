module {
  func.func @"depthwise_conv_1d_nwc_wcm_i32_1_11_4-3_4_2"() {
    %cst = arith.constant dense<[[[[160, 172], [214, 229], [280, 298], [358, 379]], [[256, 280], [334, 361], [424, 454], [526, 559]], [[352, 388], [454, 493], [568, 610], [694, 739]], [[448, 496], [574, 625], [712, 766], [862, 919]], [[544, 604], [694, 757], [856, 922], [1030, 1099]], [[640, 712], [814, 889], [1000, 1078], [1198, 1279]], [[736, 820], [934, 1021], [1144, 1234], [1366, 1459]], [[832, 928], [1054, 1153], [1288, 1390], [1534, 1639]], [[928, 1036], [1174, 1285], [1432, 1546], [1702, 1819]]]]> : tensor<1x9x4x2xi32>
    %cst_0 = arith.constant dense<[[[0, 1], [2, 3], [4, 5], [6, 7]], [[8, 9], [10, 11], [12, 13], [14, 15]], [[16, 17], [18, 19], [20, 21], [22, 23]]]> : tensor<3x4x2xi32>
    %c0_i32 = arith.constant 0 : i32
    %cst_1 = arith.constant dense<[[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23], [24, 25, 26, 27], [28, 29, 30, 31], [32, 33, 34, 35], [36, 37, 38, 39], [40, 41, 42, 43]]]> : tensor<1x11x4xi32>
    %0 = util.optimization_barrier %cst_1 : tensor<1x11x4xi32>
    %1 = util.optimization_barrier %cst_0 : tensor<3x4x2xi32>
    %2 = tensor.empty() : tensor<1x9x4x2xi32>
    %3 = linalg.fill ins(%c0_i32 : i32) outs(%2 : tensor<1x9x4x2xi32>) -> tensor<1x9x4x2xi32>
    %4 = linalg.depthwise_conv_1d_nwc_wcm {dilations = dense<1> : vector<1xi64>, strides = dense<1> : vector<1xi64>} ins(%0, %1 : tensor<1x11x4xi32>, tensor<3x4x2xi32>) outs(%3 : tensor<1x9x4x2xi32>) -> tensor<1x9x4x2xi32>
    %5 = util.optimization_barrier %cst : tensor<1x9x4x2xi32>
    "check.expect_eq"(%4, %5) : (tensor<1x9x4x2xi32>, tensor<1x9x4x2xi32>) -> ()
    return
  }
}
