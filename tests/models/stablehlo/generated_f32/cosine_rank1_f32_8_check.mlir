module {
  func.func @cosine_rank1_f32_8() {
    %cst = arith.constant dense<[1.000000e+00, 0.540302277, -0.416146785, -0.989992499, -0.653643667, 0.283662051, 0.960170209, 0.753902375]> : tensor<8xf32>
    %cst_0 = arith.constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00]> : tensor<8xf32>
    %0 = util.optimization_barrier %cst_0 : tensor<8xf32>
    %1 = stablehlo.cosine %0 : tensor<8xf32>
    %2 = util.optimization_barrier %cst : tensor<8xf32>
    check.expect_almost_eq(%1, %2, atol 5.000000e-03, rtol 9.99999997E-7) : tensor<8xf32>
    return
  }
}
