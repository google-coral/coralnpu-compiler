module {
  func.func @sine_rank1_bf16_8() {
    %cst = arith.constant dense<[0.000000e+00, 8.398430e-01, 9.101560e-01, 1.416020e-01, -7.578130e-01, -9.570310e-01, -2.792970e-01, 6.562500e-01]> : tensor<8xbf16>
    %cst_0 = arith.constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00]> : tensor<8xbf16>
    %0 = util.optimization_barrier %cst_0 : tensor<8xbf16>
    %1 = stablehlo.sine %0 : tensor<8xbf16>
    %2 = util.optimization_barrier %cst : tensor<8xbf16>
    "check.expect_almost_eq"(%1, %2) {atol = 0.00999999977 : f32, rtol = 0.00999999977 : f32} : (tensor<8xbf16>, tensor<8xbf16>) -> ()
    return
  }
}
