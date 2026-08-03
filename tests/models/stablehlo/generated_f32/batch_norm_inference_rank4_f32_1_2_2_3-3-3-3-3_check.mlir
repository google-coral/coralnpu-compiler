module {
  func.func @"batch_norm_inference_rank4_f32_1_2_2_3-3-3-3-3"() {
    %cst = arith.constant dense<[[[[0.000000e+00, 1.000000e+00, 2.000000e+00], [0.000000e+00, 3.9985013, 6.24158049]], [[0.000000e+00, 6.9970026, 10.483161], [0.000000e+00, 9.99550342, 14.7247419]]]]> : tensor<1x2x2x3xf32>
    %cst_0 = arith.constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00]> : tensor<3xf32>
    %cst_1 = arith.constant dense<[[[[0.000000e+00, 1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00, 5.000000e+00]], [[6.000000e+00, 7.000000e+00, 8.000000e+00], [9.000000e+00, 1.000000e+01, 1.100000e+01]]]]> : tensor<1x2x2x3xf32>
    %0 = util.optimization_barrier %cst_1 : tensor<1x2x2x3xf32>
    %1 = util.optimization_barrier %cst_0 : tensor<3xf32>
    %2 = util.optimization_barrier %cst_0 : tensor<3xf32>
    %3 = util.optimization_barrier %cst_0 : tensor<3xf32>
    %4 = util.optimization_barrier %cst_0 : tensor<3xf32>
    %5 = "stablehlo.batch_norm_inference"(%0, %1, %2, %3, %4) <{epsilon = 1.000000e-03 : f32, feature_index = 3 : i64}> : (tensor<1x2x2x3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>) -> tensor<1x2x2x3xf32>
    %6 = util.optimization_barrier %cst : tensor<1x2x2x3xf32>
    "check.expect_almost_eq"(%5, %6) {rtol = 9.99999997E-7 : f32} : (tensor<1x2x2x3xf32>, tensor<1x2x2x3xf32>) -> ()
    return
  }
}
