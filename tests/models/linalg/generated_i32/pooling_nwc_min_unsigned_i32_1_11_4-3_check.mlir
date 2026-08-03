module {
  func.func @"pooling_nwc_min_unsigned_i32_1_11_4-3"() {
    %cst = arith.constant dense<[[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23], [24, 25, 26, 27], [28, 29, 30, 31], [32, 33, 34, 35]]]> : tensor<1x9x4xi32>
    %cst_0 = arith.constant dense<[0, 1, 2]> : tensor<3xi32>
    %c-1_i32 = arith.constant -1 : i32
    %cst_1 = arith.constant dense<[[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23], [24, 25, 26, 27], [28, 29, 30, 31], [32, 33, 34, 35], [36, 37, 38, 39], [40, 41, 42, 43]]]> : tensor<1x11x4xi32>
    %0 = util.optimization_barrier %cst_1 : tensor<1x11x4xi32>
    %1 = util.optimization_barrier %cst_0 : tensor<3xi32>
    %2 = tensor.empty() : tensor<1x9x4xi32>
    %3 = linalg.fill ins(%c-1_i32 : i32) outs(%2 : tensor<1x9x4xi32>) -> tensor<1x9x4xi32>
    %4 = linalg.pooling_nwc_min_unsigned {dilations = dense<1> : vector<1xi64>, strides = dense<1> : vector<1xi64>} ins(%0, %1 : tensor<1x11x4xi32>, tensor<3xi32>) outs(%3 : tensor<1x9x4xi32>) -> tensor<1x9x4xi32>
    %5 = util.optimization_barrier %cst : tensor<1x9x4xi32>
    "check.expect_eq"(%4, %5) : (tensor<1x9x4xi32>, tensor<1x9x4xi32>) -> ()
    return
  }
}
