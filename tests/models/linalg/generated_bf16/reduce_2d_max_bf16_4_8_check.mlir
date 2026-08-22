module {
  func.func @reduce_2d_max_bf16_4_8() {
    %cst = arith.constant dense<[7.000000e+00, 1.500000e+01, 2.300000e+01, 3.100000e+01]> : tensor<4xbf16>
    %cst_0 = arith.constant 0xFF80 : bf16
    %cst_1 = arith.constant dense<[[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00], [8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01], [1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01], [2.400000e+01, 2.500000e+01, 2.600000e+01, 2.700000e+01, 2.800000e+01, 2.900000e+01, 3.000000e+01, 3.100000e+01]]> : tensor<4x8xbf16>
    %0 = util.optimization_barrier %cst_1 : tensor<4x8xbf16>
    %1 = tensor.empty() : tensor<4xbf16>
    %2 = linalg.fill ins(%cst_0 : bf16) outs(%1 : tensor<4xbf16>) -> tensor<4xbf16>
    %reduced = linalg.reduce ins(%0 : tensor<4x8xbf16>) outs(%2 : tensor<4xbf16>) dimensions = [1] 
      (%in: bf16, %init: bf16) {
        %4 = arith.maximumf %in, %init : bf16
        linalg.yield %4 : bf16
      }
    %3 = util.optimization_barrier %cst : tensor<4xbf16>
    "check.expect_almost_eq"(%reduced, %3) {atol = 0.00999999977 : f32, rtol = 0.00999999977 : f32} : (tensor<4xbf16>, tensor<4xbf16>) -> ()
    return
  }
}
