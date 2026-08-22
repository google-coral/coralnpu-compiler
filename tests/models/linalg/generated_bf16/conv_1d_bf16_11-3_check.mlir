module {
  func.func @"conv_1d_bf16_11-3"() {
    %cst = arith.constant dense<[5.000000e+00, 8.000000e+00, 1.100000e+01, 1.400000e+01, 1.700000e+01, 2.000000e+01, 2.300000e+01, 2.600000e+01, 2.900000e+01]> : tensor<9xbf16>
    %cst_0 = arith.constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00]> : tensor<3xbf16>
    %cst_1 = arith.constant 0.000000e+00 : bf16
    %cst_2 = arith.constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01]> : tensor<11xbf16>
    %0 = util.optimization_barrier %cst_2 : tensor<11xbf16>
    %1 = util.optimization_barrier %cst_0 : tensor<3xbf16>
    %2 = tensor.empty() : tensor<9xbf16>
    %3 = linalg.fill ins(%cst_1 : bf16) outs(%2 : tensor<9xbf16>) -> tensor<9xbf16>
    %4 = linalg.conv_1d ins(%0, %1 : tensor<11xbf16>, tensor<3xbf16>) outs(%3 : tensor<9xbf16>) -> tensor<9xbf16>
    %5 = util.optimization_barrier %cst : tensor<9xbf16>
    "check.expect_almost_eq"(%4, %5) {atol = 0.00999999977 : f32, rtol = 0.00999999977 : f32} : (tensor<9xbf16>, tensor<9xbf16>) -> ()
    return
  }
}
