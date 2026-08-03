module {
  func.func @tan_rank1_f32_8() {
    %cst = arith.constant dense<[0.000000e+00, 1.55740774, -2.18504024, -0.142546564, 1.15782106, -3.38051677, -0.291006327, 0.871447682]> : tensor<8xf32>
    %cst_0 = arith.constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00]> : tensor<8xf32>
    %0 = util.optimization_barrier %cst_0 : tensor<8xf32>
    %1 = stablehlo.tan %0 : tensor<8xf32>
    %2 = util.optimization_barrier %cst : tensor<8xf32>
    "check.expect_almost_eq"(%1, %2) {rtol = 9.99999997E-7 : f32} : (tensor<8xf32>, tensor<8xf32>) -> ()
    return
  }
}
