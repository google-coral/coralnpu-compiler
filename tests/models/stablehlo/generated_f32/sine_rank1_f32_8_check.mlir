module {
  func.func @sine_rank1_f32_8() {
    %cst = arith.constant dense<[0.000000e+00, 0.841470957, 0.909297466, 0.141120031, -0.756802439, -0.958924293, -0.279415607, 0.656986475]> : tensor<8xf32>
    %cst_0 = arith.constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00]> : tensor<8xf32>
    %0 = util.optimization_barrier %cst_0 : tensor<8xf32>
    %1 = stablehlo.sine %0 : tensor<8xf32>
    %2 = util.optimization_barrier %cst : tensor<8xf32>
    check.expect_almost_eq(%1, %2, atol 5.000000e-03, rtol 9.99999997E-7) : tensor<8xf32>
    return
  }
}
