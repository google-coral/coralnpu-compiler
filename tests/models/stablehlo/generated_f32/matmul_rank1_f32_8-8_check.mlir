module {
  func.func @"matmul_rank1_f32_8-8"() {
    %cst = arith.constant dense<1.400000e+02> : tensor<f32>
    %cst_0 = arith.constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00]> : tensor<8xf32>
    %0 = util.optimization_barrier %cst_0 : tensor<8xf32>
    %1 = util.optimization_barrier %cst_0 : tensor<8xf32>
    %2 = stablehlo.dot %0, %1 : (tensor<8xf32>, tensor<8xf32>) -> tensor<f32>
    %3 = util.optimization_barrier %cst : tensor<f32>
    "check.expect_almost_eq"(%2, %3) {rtol = 9.99999997E-7 : f32} : (tensor<f32>, tensor<f32>) -> ()
    return
  }
}
