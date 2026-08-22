module {
  func.func @"batch_norm_inference_rank4_bf16_1_2_2_3-3-3-3-3"() {
    %cst = arith.constant dense<[[[[0.000000e+00, 1.000000e+00, 2.000000e+00], [0.000000e+00, 4.000000e+00, 6.250000e+00]], [[0.000000e+00, 7.000000e+00, 1.050000e+01], [0.000000e+00, 1.000000e+01, 1.475000e+01]]]]> : tensor<1x2x2x3xbf16>
    %cst_0 = arith.constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00]> : tensor<3xbf16>
    %cst_1 = arith.constant dense<[[[[0.000000e+00, 1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00, 5.000000e+00]], [[6.000000e+00, 7.000000e+00, 8.000000e+00], [9.000000e+00, 1.000000e+01, 1.100000e+01]]]]> : tensor<1x2x2x3xbf16>
    %0 = util.optimization_barrier %cst_1 : tensor<1x2x2x3xbf16>
    %1 = util.optimization_barrier %cst_0 : tensor<3xbf16>
    %2 = util.optimization_barrier %cst_0 : tensor<3xbf16>
    %3 = util.optimization_barrier %cst_0 : tensor<3xbf16>
    %4 = util.optimization_barrier %cst_0 : tensor<3xbf16>
    %5 = "stablehlo.batch_norm_inference"(%0, %1, %2, %3, %4) <{epsilon = 1.000000e-03 : f32, feature_index = 3 : i64}> : (tensor<1x2x2x3xbf16>, tensor<3xbf16>, tensor<3xbf16>, tensor<3xbf16>, tensor<3xbf16>) -> tensor<1x2x2x3xbf16>
    %6 = util.optimization_barrier %cst : tensor<1x2x2x3xbf16>
    "check.expect_almost_eq"(%5, %6) {atol = 0.00999999977 : f32, rtol = 0.00999999977 : f32} : (tensor<1x2x2x3xbf16>, tensor<1x2x2x3xbf16>) -> ()
    return
  }
}
