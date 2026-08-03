module {
  func.func @"div_f32_4_8-4_8"() {
    %cst = arith.constant dense<1.000000e+00> : tensor<4x8xf32>
    %cst_0 = arith.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 1.000000e+00], [2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00], [5.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 1.000000e+00, 2.000000e+00]]> : tensor<4x8xf32>
    %0 = util.optimization_barrier %cst_0 : tensor<4x8xf32>
    %1 = util.optimization_barrier %cst_0 : tensor<4x8xf32>
    %2 = tensor.empty() : tensor<4x8xf32>
    %3 = linalg.div ins(%0, %1 : tensor<4x8xf32>, tensor<4x8xf32>) outs(%2 : tensor<4x8xf32>) -> tensor<4x8xf32>
    %4 = util.optimization_barrier %cst : tensor<4x8xf32>
    "check.expect_almost_eq"(%3, %4) {rtol = 9.99999997E-7 : f32} : (tensor<4x8xf32>, tensor<4x8xf32>) -> ()
    return
  }
}
