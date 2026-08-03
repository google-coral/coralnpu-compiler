module {
  func.func @exponential_rank1_f32_8() {
    %cst = arith.constant dense<[1.000000e+00, 2.71828175, 7.3890562, 20.085537, 54.5981522, 148.413162, 403.428802, 1096.63318]> : tensor<8xf32>
    %cst_0 = arith.constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00]> : tensor<8xf32>
    %0 = util.optimization_barrier %cst_0 : tensor<8xf32>
    %1 = stablehlo.exponential %0 : tensor<8xf32>
    %2 = util.optimization_barrier %cst : tensor<8xf32>
    "check.expect_almost_eq"(%1, %2) {rtol = 9.99999997E-7 : f32} : (tensor<8xf32>, tensor<8xf32>) -> ()
    return
  }
}
