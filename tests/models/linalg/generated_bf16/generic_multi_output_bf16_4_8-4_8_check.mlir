#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @"generic_multi_output_bf16_4_8-4_8"() {
    %cst = arith.constant dense<0.000000e+00> : tensor<4x8xbf16>
    %cst_0 = arith.constant dense<[[0.000000e+00, 2.000000e+00, 4.000000e+00, 6.000000e+00, 8.000000e+00, 1.000000e+01, 1.200000e+01, 1.400000e+01], [1.600000e+01, 1.800000e+01, 2.000000e+01, 2.200000e+01, 2.400000e+01, 2.600000e+01, 2.800000e+01, 3.000000e+01], [3.200000e+01, 3.400000e+01, 3.600000e+01, 3.800000e+01, 4.000000e+01, 4.200000e+01, 4.400000e+01, 4.600000e+01], [4.800000e+01, 5.000000e+01, 5.200000e+01, 5.400000e+01, 5.600000e+01, 5.800000e+01, 6.000000e+01, 6.200000e+01]]> : tensor<4x8xbf16>
    %cst_1 = arith.constant dense<[[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00], [8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01], [1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01], [2.400000e+01, 2.500000e+01, 2.600000e+01, 2.700000e+01, 2.800000e+01, 2.900000e+01, 3.000000e+01, 3.100000e+01]]> : tensor<4x8xbf16>
    %0 = util.optimization_barrier %cst_1 : tensor<4x8xbf16>
    %1 = util.optimization_barrier %cst_1 : tensor<4x8xbf16>
    %2 = tensor.empty() : tensor<4x8xbf16>
    %3:2 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%0, %1 : tensor<4x8xbf16>, tensor<4x8xbf16>) outs(%2, %2 : tensor<4x8xbf16>, tensor<4x8xbf16>) {
    ^bb0(%in: bf16, %in_2: bf16, %out: bf16, %out_3: bf16):
      %6 = arith.addf %in, %in_2 : bf16
      %7 = arith.subf %in, %in_2 : bf16
      linalg.yield %6, %7 : bf16, bf16
    } -> (tensor<4x8xbf16>, tensor<4x8xbf16>)
    %4 = util.optimization_barrier %cst_0 : tensor<4x8xbf16>
    "check.expect_almost_eq"(%3#0, %4) {atol = 0.00999999977 : f32, rtol = 0.00999999977 : f32} : (tensor<4x8xbf16>, tensor<4x8xbf16>) -> ()
    %5 = util.optimization_barrier %cst : tensor<4x8xbf16>
    "check.expect_almost_eq"(%3#1, %5) {atol = 0.00999999977 : f32, rtol = 0.00999999977 : f32} : (tensor<4x8xbf16>, tensor<4x8xbf16>) -> ()
    return
  }
}
