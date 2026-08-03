module {
  func.func @"pooling_nwc_max_i16_1_11_4-3"() {
    %cst = arith.constant dense<[[[8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23], [24, 25, 26, 27], [28, 29, 30, 31], [32, 33, 34, 35], [36, 37, 38, 39], [40, 41, 42, 43]]]> : tensor<1x9x4xi16>
    %cst_0 = arith.constant dense<[0, 1, 2]> : tensor<3xi16>
    %c-32768_i16 = arith.constant -32768 : i16
    %cst_1 = arith.constant dense<[[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23], [24, 25, 26, 27], [28, 29, 30, 31], [32, 33, 34, 35], [36, 37, 38, 39], [40, 41, 42, 43]]]> : tensor<1x11x4xi16>
    %0 = util.optimization_barrier %cst_1 : tensor<1x11x4xi16>
    %1 = util.optimization_barrier %cst_0 : tensor<3xi16>
    %2 = tensor.empty() : tensor<1x9x4xi16>
    %3 = linalg.fill ins(%c-32768_i16 : i16) outs(%2 : tensor<1x9x4xi16>) -> tensor<1x9x4xi16>
    %4 = linalg.pooling_nwc_max {dilations = dense<1> : vector<1xi64>, strides = dense<1> : vector<1xi64>} ins(%0, %1 : tensor<1x11x4xi16>, tensor<3xi16>) outs(%3 : tensor<1x9x4xi16>) -> tensor<1x9x4xi16>
    %5 = util.optimization_barrier %cst : tensor<1x9x4xi16>
    "check.expect_eq"(%4, %5) : (tensor<1x9x4xi16>, tensor<1x9x4xi16>) -> ()
    return
  }
}
