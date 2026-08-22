module {
  func.func @"vecmat_bf16_8-8_4"() {
    %cst = arith.constant dense<[5.600000e+02, 5.880000e+02, 6.160000e+02, 6.440000e+02]> : tensor<4xbf16>
    %cst_0 = arith.constant dense<[[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00], [8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01], [1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01], [1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01], [2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01], [2.400000e+01, 2.500000e+01, 2.600000e+01, 2.700000e+01], [2.800000e+01, 2.900000e+01, 3.000000e+01, 3.100000e+01]]> : tensor<8x4xbf16>
    %cst_1 = arith.constant 0.000000e+00 : bf16
    %cst_2 = arith.constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00]> : tensor<8xbf16>
    %0 = util.optimization_barrier %cst_2 : tensor<8xbf16>
    %1 = util.optimization_barrier %cst_0 : tensor<8x4xbf16>
    %2 = tensor.empty() : tensor<4xbf16>
    %3 = linalg.fill ins(%cst_1 : bf16) outs(%2 : tensor<4xbf16>) -> tensor<4xbf16>
    %4 = linalg.vecmat ins(%0, %1 : tensor<8xbf16>, tensor<8x4xbf16>) outs(%3 : tensor<4xbf16>) -> tensor<4xbf16>
    %5 = util.optimization_barrier %cst : tensor<4xbf16>
    "check.expect_almost_eq"(%4, %5) {atol = 0.00999999977 : f32, rtol = 0.00999999977 : f32} : (tensor<4xbf16>, tensor<4xbf16>) -> ()
    return
  }
}
