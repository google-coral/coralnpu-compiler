module {
  func.func @"dot_f32_8-8"() {
    %cst = arith.constant dense<1.400000e+02> : tensor<f32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00]> : tensor<8xf32>
    %0 = util.optimization_barrier %cst_1 : tensor<8xf32>
    %1 = util.optimization_barrier %cst_1 : tensor<8xf32>
    %2 = tensor.empty() : tensor<f32>
    %3 = linalg.fill ins(%cst_0 : f32) outs(%2 : tensor<f32>) -> tensor<f32>
    %4 = linalg.dot ins(%0, %1 : tensor<8xf32>, tensor<8xf32>) outs(%3 : tensor<f32>) -> tensor<f32>
    %5 = util.optimization_barrier %cst : tensor<f32>
    check.expect_almost_eq(%4, %5, atol 1.010000e-04, rtol 2.000000e-06) : tensor<f32>
    return
  }
}
