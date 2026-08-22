module {
  func.func @"if_rank1_bf16_1-8"() {
    %cst = arith.constant dense<[-0.000000e+00, -1.000000e+00, -2.000000e+00, -3.000000e+00, -4.000000e+00, -5.000000e+00, -6.000000e+00, -7.000000e+00]> : tensor<8xbf16>
    %cst_0 = arith.constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00]> : tensor<8xbf16>
    %cst_1 = arith.constant dense<false> : tensor<1xi1>
    %0 = util.optimization_barrier %cst_1 : tensor<1xi1>
    %1 = util.optimization_barrier %cst_0 : tensor<8xbf16>
    %2 = stablehlo.reshape %0 : (tensor<1xi1>) -> tensor<i1>
    %3 = "stablehlo.if"(%2) ({
      stablehlo.return %1 : tensor<8xbf16>
    }, {
      %5 = stablehlo.negate %1 : tensor<8xbf16>
      stablehlo.return %5 : tensor<8xbf16>
    }) : (tensor<i1>) -> tensor<8xbf16>
    %4 = util.optimization_barrier %cst : tensor<8xbf16>
    "check.expect_almost_eq"(%3, %4) {atol = 0.00999999977 : f32, rtol = 0.00999999977 : f32} : (tensor<8xbf16>, tensor<8xbf16>) -> ()
    return
  }
}
