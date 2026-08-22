module {
  func.func @sort_rank1_bf16_8() {
    %cst = arith.constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00]> : tensor<8xbf16>
    %0 = util.optimization_barrier %cst : tensor<8xbf16>
    %1 = "stablehlo.sort"(%0) <{dimension = 0 : i64, is_stable = false}> ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %3 = stablehlo.compare LT, %arg0, %arg1 : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
      stablehlo.return %3 : tensor<i1>
    }) : (tensor<8xbf16>) -> tensor<8xbf16>
    %2 = util.optimization_barrier %cst : tensor<8xbf16>
    "check.expect_almost_eq"(%1, %2) {atol = 0.00999999977 : f32, rtol = 0.00999999977 : f32} : (tensor<8xbf16>, tensor<8xbf16>) -> ()
    return
  }
}
