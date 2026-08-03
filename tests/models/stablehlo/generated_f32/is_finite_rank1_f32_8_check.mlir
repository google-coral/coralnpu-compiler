module {
  func.func @is_finite_rank1_f32_8() {
    %cst = arith.constant dense<true> : tensor<8xi1>
    %cst_0 = arith.constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00]> : tensor<8xf32>
    %0 = util.optimization_barrier %cst_0 : tensor<8xf32>
    %1 = stablehlo.is_finite %0 : (tensor<8xf32>) -> tensor<8xi1>
    %2 = util.optimization_barrier %cst : tensor<8xi1>
    "check.expect_eq"(%1, %2) : (tensor<8xi1>, tensor<8xi1>) -> ()
    return
  }
}
