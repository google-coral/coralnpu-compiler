module {
  func.func @"conv_1d_nwc_wcf_i8_1_11_4-3_4_8"() {
    %cst = arith.constant dense<[[[-48, 18, 84, -106, -40, 26, 92, -98], [16, -126, -12, 102, -40, 74, -68, 46], [80, -14, -108, 54, -40, 122, 28, -66], [-112, 98, 52, 6, -40, -86, 124, 78], [-48, -46, -44, -42, -40, -38, -36, -34], [16, 66, 116, -90, -40, 10, 60, 110], [80, -78, 20, 118, -40, 58, -100, -2], [-112, 34, -76, 70, -40, 106, -4, -114], [-48, -110, 84, 22, -40, -102, 92, 30]]]> : tensor<1x9x8xi8>
    %cst_0 = arith.constant dense<[[[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15], [16, 17, 18, 19, 20, 21, 22, 23], [24, 25, 26, 27, 28, 29, 30, 31]], [[32, 33, 34, 35, 36, 37, 38, 39], [40, 41, 42, 43, 44, 45, 46, 47], [48, 49, 50, 51, 52, 53, 54, 55], [56, 57, 58, 59, 60, 61, 62, 63]], [[64, 65, 66, 67, 68, 69, 70, 71], [72, 73, 74, 75, 76, 77, 78, 79], [80, 81, 82, 83, 84, 85, 86, 87], [88, 89, 90, 91, 92, 93, 94, 95]]]> : tensor<3x4x8xi8>
    %c0_i8 = arith.constant 0 : i8
    %cst_1 = arith.constant dense<[[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23], [24, 25, 26, 27], [28, 29, 30, 31], [32, 33, 34, 35], [36, 37, 38, 39], [40, 41, 42, 43]]]> : tensor<1x11x4xi8>
    %0 = util.optimization_barrier %cst_1 : tensor<1x11x4xi8>
    %1 = util.optimization_barrier %cst_0 : tensor<3x4x8xi8>
    %2 = tensor.empty() : tensor<1x9x8xi8>
    %3 = linalg.fill ins(%c0_i8 : i8) outs(%2 : tensor<1x9x8xi8>) -> tensor<1x9x8xi8>
    %4 = linalg.conv_1d_nwc_wcf {dilations = dense<1> : vector<1xi64>, strides = dense<1> : vector<1xi64>} ins(%0, %1 : tensor<1x11x4xi8>, tensor<3x4x8xi8>) outs(%3 : tensor<1x9x8xi8>) -> tensor<1x9x8xi8>
    %5 = util.optimization_barrier %cst : tensor<1x9x8xi8>
    "check.expect_eq"(%4, %5) : (tensor<1x9x8xi8>, tensor<1x9x8xi8>) -> ()
    return
  }
}
