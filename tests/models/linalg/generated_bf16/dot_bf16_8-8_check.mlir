module {
  func.func @"dot_bf16_8-8"() {
    %cst = arith.constant dense<1.400000e+02> : tensor<bf16>
    %cst_0 = arith.constant 0.000000e+00 : bf16
    %cst_1 = arith.constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00]> : tensor<8xbf16>
    %0 = util.optimization_barrier %cst_1 : tensor<8xbf16>
    %1 = util.optimization_barrier %cst_1 : tensor<8xbf16>
    %2 = tensor.empty() : tensor<bf16>
    %3 = linalg.fill ins(%cst_0 : bf16) outs(%2 : tensor<bf16>) -> tensor<bf16>
    %4 = linalg.dot ins(%0, %1 : tensor<8xbf16>, tensor<8xbf16>) outs(%3 : tensor<bf16>) -> tensor<bf16>
    %5 = util.optimization_barrier %cst : tensor<bf16>
    "check.expect_almost_eq"(%4, %5) {atol = 0.00999999977 : f32, rtol = 0.00999999977 : f32} : (tensor<bf16>, tensor<bf16>) -> ()
    return
  }
}
