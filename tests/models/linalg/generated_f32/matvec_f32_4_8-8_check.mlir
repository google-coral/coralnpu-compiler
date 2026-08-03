module {
  func.func @"matvec_f32_4_8-8"() {
    %cst = arith.constant dense<[1.400000e+02, 3.640000e+02, 5.880000e+02, 8.120000e+02]> : tensor<4xf32>
    %cst_0 = arith.constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00]> : tensor<8xf32>
    %cst_1 = arith.constant 0.000000e+00 : f32
    %cst_2 = arith.constant dense<[[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00], [8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01], [1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01], [2.400000e+01, 2.500000e+01, 2.600000e+01, 2.700000e+01, 2.800000e+01, 2.900000e+01, 3.000000e+01, 3.100000e+01]]> : tensor<4x8xf32>
    %0 = util.optimization_barrier %cst_2 : tensor<4x8xf32>
    %1 = util.optimization_barrier %cst_0 : tensor<8xf32>
    %2 = tensor.empty() : tensor<4xf32>
    %3 = linalg.fill ins(%cst_1 : f32) outs(%2 : tensor<4xf32>) -> tensor<4xf32>
    %4 = linalg.matvec ins(%0, %1 : tensor<4x8xf32>, tensor<8xf32>) outs(%3 : tensor<4xf32>) -> tensor<4xf32>
    %5 = util.optimization_barrier %cst : tensor<4xf32>
    "check.expect_almost_eq"(%4, %5) {rtol = 9.99999997E-7 : f32} : (tensor<4xf32>, tensor<4xf32>) -> ()
    return
  }
}
