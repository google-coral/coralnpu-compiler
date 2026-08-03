module {
  func.func @reduce_sum_rank1_f32_8() {
    %cst = arith.constant dense<2.800000e+01> : tensor<f32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_1 = arith.constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00]> : tensor<8xf32>
    %0 = util.optimization_barrier %cst_1 : tensor<8xf32>
    %1 = stablehlo.reduce(%0 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<8xf32>, tensor<f32>) -> tensor<f32>
    %2 = util.optimization_barrier %cst : tensor<f32>
    "check.expect_almost_eq"(%1, %2) {rtol = 9.99999997E-7 : f32} : (tensor<f32>, tensor<f32>) -> ()
    return
  }
}
