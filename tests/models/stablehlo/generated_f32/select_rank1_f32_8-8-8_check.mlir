module {
  func.func @"select_rank1_f32_8-8-8"() {
    %cst = arith.constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00]> : tensor<8xf32>
    %cst_0 = arith.constant dense<[false, true, false, true, false, true, false, true]> : tensor<8xi1>
    %0 = util.optimization_barrier %cst_0 : tensor<8xi1>
    %1 = util.optimization_barrier %cst : tensor<8xf32>
    %2 = util.optimization_barrier %cst : tensor<8xf32>
    %3 = stablehlo.select %0, %1, %2 : tensor<8xi1>, tensor<8xf32>
    %4 = util.optimization_barrier %cst : tensor<8xf32>
    "check.expect_almost_eq"(%3, %4) {rtol = 9.99999997E-7 : f32} : (tensor<8xf32>, tensor<8xf32>) -> ()
    return
  }
}
