module {
  func.func @reduce_sum_rank3_f32_2_3_4() {
    %cst = arith.constant dense<[[6.000000e+00, 2.200000e+01, 3.800000e+01], [5.400000e+01, 7.000000e+01, 8.600000e+01]]> : tensor<2x3xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_1 = arith.constant dense<[[[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00], [8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01]], [[1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01], [1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01], [2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01]]]> : tensor<2x3x4xf32>
    %0 = util.optimization_barrier %cst_1 : tensor<2x3x4xf32>
    %1 = stablehlo.reduce(%0 init: %cst_0) applies stablehlo.add across dimensions = [2] : (tensor<2x3x4xf32>, tensor<f32>) -> tensor<2x3xf32>
    %2 = util.optimization_barrier %cst : tensor<2x3xf32>
    "check.expect_almost_eq"(%1, %2) {rtol = 9.99999997E-7 : f32} : (tensor<2x3xf32>, tensor<2x3xf32>) -> ()
    return
  }
}
