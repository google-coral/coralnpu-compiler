module {
  func.func @"pooling_nwc_min_unsigned_i8_1_11_4-3"() {
    %cst = arith.constant dense<[[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23], [24, 25, 26, 27], [28, 29, 30, 31], [32, 33, 34, 35]]]> : tensor<1x9x4xi8>
    %cst_0 = arith.constant dense<[0, 1, 2]> : tensor<3xi8>
    %c-1_i8 = arith.constant -1 : i8
    %cst_1 = arith.constant dense<[[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23], [24, 25, 26, 27], [28, 29, 30, 31], [32, 33, 34, 35], [36, 37, 38, 39], [40, 41, 42, 43]]]> : tensor<1x11x4xi8>
    %0 = util.optimization_barrier %cst_1 : tensor<1x11x4xi8>
    %1 = util.optimization_barrier %cst_0 : tensor<3xi8>
    %2 = tensor.empty() : tensor<1x9x4xi8>
    %3 = linalg.fill ins(%c-1_i8 : i8) outs(%2 : tensor<1x9x4xi8>) -> tensor<1x9x4xi8>
    %4 = linalg.pooling_nwc_min_unsigned {dilations = dense<1> : vector<1xi64>, strides = dense<1> : vector<1xi64>} ins(%0, %1 : tensor<1x11x4xi8>, tensor<3xi8>) outs(%3 : tensor<1x9x4xi8>) -> tensor<1x9x4xi8>
    %5 = util.optimization_barrier %cst : tensor<1x9x4xi8>
    "check.expect_eq"(%4, %5) : (tensor<1x9x4xi8>, tensor<1x9x4xi8>) -> ()
    return
  }
}
