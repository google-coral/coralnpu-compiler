#map = affine_map<(d0) -> (d0)>
module {
  func.func @"generic_add_rank1_bf16_8-8"() {
    %cst = arith.constant dense<[0.000000e+00, 2.000000e+00, 4.000000e+00, 6.000000e+00, 8.000000e+00, 1.000000e+01, 1.200000e+01, 1.400000e+01]> : tensor<8xbf16>
    %cst_0 = arith.constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00]> : tensor<8xbf16>
    %0 = util.optimization_barrier %cst_0 : tensor<8xbf16>
    %1 = util.optimization_barrier %cst_0 : tensor<8xbf16>
    %2 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%0, %1 : tensor<8xbf16>, tensor<8xbf16>) outs(%0 : tensor<8xbf16>) {
    ^bb0(%in: bf16, %in_1: bf16, %out: bf16):
      %4 = arith.addf %in, %in_1 : bf16
      linalg.yield %4 : bf16
    } -> tensor<8xbf16>
    %3 = util.optimization_barrier %cst : tensor<8xbf16>
    "check.expect_almost_eq"(%2, %3) {atol = 0.00999999977 : f32, rtol = 0.00999999977 : f32} : (tensor<8xbf16>, tensor<8xbf16>) -> ()
    return
  }
}
