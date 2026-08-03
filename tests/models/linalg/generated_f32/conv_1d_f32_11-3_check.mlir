module {
  func.func @"conv_1d_f32_11-3"() {
    %cst = arith.constant dense<[5.000000e+00, 8.000000e+00, 1.100000e+01, 1.400000e+01, 1.700000e+01, 2.000000e+01, 2.300000e+01, 2.600000e+01, 2.900000e+01]> : tensor<9xf32>
    %cst_0 = arith.constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00]> : tensor<3xf32>
    %cst_1 = arith.constant 0.000000e+00 : f32
    %cst_2 = arith.constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01]> : tensor<11xf32>
    %0 = util.optimization_barrier %cst_2 : tensor<11xf32>
    %1 = util.optimization_barrier %cst_0 : tensor<3xf32>
    %2 = tensor.empty() : tensor<9xf32>
    %3 = linalg.fill ins(%cst_1 : f32) outs(%2 : tensor<9xf32>) -> tensor<9xf32>
    %4 = linalg.conv_1d ins(%0, %1 : tensor<11xf32>, tensor<3xf32>) outs(%3 : tensor<9xf32>) -> tensor<9xf32>
    %5 = util.optimization_barrier %cst : tensor<9xf32>
    "check.expect_almost_eq"(%4, %5) {rtol = 9.99999997E-7 : f32} : (tensor<9xf32>, tensor<9xf32>) -> ()
    return
  }
}
