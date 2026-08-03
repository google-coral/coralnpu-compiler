module {
  func.func @cbrt_rank1_f32_8() {
    %cst = arith.constant dense<[0.000000e+00, 1.00000012, 1.25992239, 1.44224977, 1.58740211, 1.70997596, 1.81712055, 1.91293132]> : tensor<8xf32>
    %cst_0 = arith.constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00]> : tensor<8xf32>
    %0 = util.optimization_barrier %cst_0 : tensor<8xf32>
    %1 = stablehlo.cbrt %0 : tensor<8xf32>
    %2 = util.optimization_barrier %cst : tensor<8xf32>
    "check.expect_almost_eq"(%1, %2) {rtol = 9.99999997E-7 : f32} : (tensor<8xf32>, tensor<8xf32>) -> ()
    return
  }
}
