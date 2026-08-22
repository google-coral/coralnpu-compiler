module {
  func.func @fill_rank1_bf16_8() {
    %cst = arith.constant dense<1.000000e+00> : tensor<8xbf16>
    %cst_0 = arith.constant 1.000000e+00 : bf16
    %cst_1 = arith.constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00]> : tensor<8xbf16>
    %0 = util.optimization_barrier %cst_1 : tensor<8xbf16>
    %1 = linalg.fill ins(%cst_0 : bf16) outs(%0 : tensor<8xbf16>) -> tensor<8xbf16>
    %2 = util.optimization_barrier %cst : tensor<8xbf16>
    "check.expect_almost_eq"(%1, %2) {atol = 0.00999999977 : f32, rtol = 0.00999999977 : f32} : (tensor<8xbf16>, tensor<8xbf16>) -> ()
    return
  }
}
