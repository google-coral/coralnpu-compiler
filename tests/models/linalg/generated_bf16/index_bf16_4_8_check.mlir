#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @index_bf16_4_8() {
    %cst = arith.constant dense<[[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00], [1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00], [2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00], [3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01]]> : tensor<4x8xbf16>
    %cst_0 = arith.constant dense<[[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00], [8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01], [1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01], [2.400000e+01, 2.500000e+01, 2.600000e+01, 2.700000e+01, 2.800000e+01, 2.900000e+01, 3.000000e+01, 3.100000e+01]]> : tensor<4x8xbf16>
    %0 = util.optimization_barrier %cst_0 : tensor<4x8xbf16>
    %1 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%0 : tensor<4x8xbf16>) {
    ^bb0(%out: bf16):
      %3 = linalg.index 0 : index
      %4 = linalg.index 1 : index
      %5 = arith.index_cast %3 : index to i32
      %6 = arith.index_cast %4 : index to i32
      %7 = arith.sitofp %5 : i32 to bf16
      %8 = arith.sitofp %6 : i32 to bf16
      %9 = arith.addf %7, %8 : bf16
      linalg.yield %9 : bf16
    } -> tensor<4x8xbf16>
    %2 = util.optimization_barrier %cst : tensor<4x8xbf16>
    "check.expect_almost_eq"(%1, %2) {atol = 0.00999999977 : f32, rtol = 0.00999999977 : f32} : (tensor<4x8xbf16>, tensor<4x8xbf16>) -> ()
    return
  }
}
