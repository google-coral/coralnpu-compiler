#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  func.func @"generic_add_rank3_bf16_2_3_4-2_3_4"() {
    %cst = arith.constant dense<[[[0.000000e+00, 2.000000e+00, 4.000000e+00, 6.000000e+00], [8.000000e+00, 1.000000e+01, 1.200000e+01, 1.400000e+01], [1.600000e+01, 1.800000e+01, 2.000000e+01, 2.200000e+01]], [[2.400000e+01, 2.600000e+01, 2.800000e+01, 3.000000e+01], [3.200000e+01, 3.400000e+01, 3.600000e+01, 3.800000e+01], [4.000000e+01, 4.200000e+01, 4.400000e+01, 4.600000e+01]]]> : tensor<2x3x4xbf16>
    %cst_0 = arith.constant dense<[[[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00], [8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01]], [[1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01], [1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01], [2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01]]]> : tensor<2x3x4xbf16>
    %0 = util.optimization_barrier %cst_0 : tensor<2x3x4xbf16>
    %1 = util.optimization_barrier %cst_0 : tensor<2x3x4xbf16>
    %2 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0, %1 : tensor<2x3x4xbf16>, tensor<2x3x4xbf16>) outs(%0 : tensor<2x3x4xbf16>) {
    ^bb0(%in: bf16, %in_1: bf16, %out: bf16):
      %4 = arith.addf %in, %in_1 : bf16
      linalg.yield %4 : bf16
    } -> tensor<2x3x4xbf16>
    %3 = util.optimization_barrier %cst : tensor<2x3x4xbf16>
    "check.expect_almost_eq"(%2, %3) {atol = 0.00999999977 : f32, rtol = 0.00999999977 : f32} : (tensor<2x3x4xbf16>, tensor<2x3x4xbf16>) -> ()
    return
  }
}
