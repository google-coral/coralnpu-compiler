module {
  func.func @sort_rank1_f32_8() {
    %cst = arith.constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00]> : tensor<8xf32>
    %0 = util.optimization_barrier %cst : tensor<8xf32>
    %1 = "stablehlo.sort"(%0) <{dimension = 0 : i64, is_stable = false}> ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %3 = stablehlo.compare LT, %arg0, %arg1 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      stablehlo.return %3 : tensor<i1>
    }) : (tensor<8xf32>) -> tensor<8xf32>
    %2 = util.optimization_barrier %cst : tensor<8xf32>
    "check.expect_almost_eq"(%1, %2) {rtol = 9.99999997E-7 : f32} : (tensor<8xf32>, tensor<8xf32>) -> ()
    return
  }
}
