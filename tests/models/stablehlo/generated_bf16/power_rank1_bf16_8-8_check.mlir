module {
  func.func @"power_rank1_bf16_8-8"() {
    %cst = arith.constant dense<[0x7FC0, 1.000000e+00, 4.000000e+00, 2.700000e+01, 2.560000e+02, 3.120000e+03, 4.659200e+04, 8.232960e+05]> : tensor<8xbf16>
    %cst_0 = arith.constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00]> : tensor<8xbf16>
    %0 = util.optimization_barrier %cst_0 : tensor<8xbf16>
    %1 = util.optimization_barrier %cst_0 : tensor<8xbf16>
    %2 = stablehlo.power %0, %1 : tensor<8xbf16>
    %3 = util.optimization_barrier %cst : tensor<8xbf16>
    "check.expect_almost_eq"(%2, %3) {atol = 0.00999999977 : f32, rtol = 0.00999999977 : f32} : (tensor<8xbf16>, tensor<8xbf16>) -> ()
    return
  }
}
