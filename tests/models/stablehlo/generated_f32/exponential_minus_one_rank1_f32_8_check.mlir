module {
  func.func @exponential_minus_one_rank1_f32_8() {
    %cst = arith.constant dense<[0.000000e+00, 1.71828175, 6.3890562, 19.085537, 53.5981522, 147.413162, 402.428802, 1095.63318]> : tensor<8xf32>
    %cst_0 = arith.constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00]> : tensor<8xf32>
    %0 = util.optimization_barrier %cst_0 : tensor<8xf32>
    %1 = stablehlo.exponential_minus_one %0 : tensor<8xf32>
    %2 = util.optimization_barrier %cst : tensor<8xf32>
    "check.expect_almost_eq"(%1, %2) {rtol = 9.99999997E-7 : f32} : (tensor<8xf32>, tensor<8xf32>) -> ()
    return
  }
}
