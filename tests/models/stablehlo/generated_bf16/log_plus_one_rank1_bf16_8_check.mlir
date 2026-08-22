module {
  func.func @log_plus_one_rank1_bf16_8() {
    %cst = arith.constant dense<[0.000000e+00, 6.914060e-01, 1.101560e+00, 1.382810e+00, 1.609380e+00, 1.789060e+00, 1.945310e+00, 2.078130e+00]> : tensor<8xbf16>
    %cst_0 = arith.constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00]> : tensor<8xbf16>
    %0 = util.optimization_barrier %cst_0 : tensor<8xbf16>
    %1 = stablehlo.log_plus_one %0 : tensor<8xbf16>
    %2 = util.optimization_barrier %cst : tensor<8xbf16>
    "check.expect_almost_eq"(%1, %2) {atol = 0.00999999977 : f32, rtol = 0.00999999977 : f32} : (tensor<8xbf16>, tensor<8xbf16>) -> ()
    return
  }
}
