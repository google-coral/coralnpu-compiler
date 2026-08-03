#map = affine_map<(d0) -> (d0)>
module {
  func.func @"generic_add_rank1_f32_8-8"() {
    %cst = arith.constant dense<[0.000000e+00, 2.000000e+00, 4.000000e+00, 6.000000e+00, 8.000000e+00, 1.000000e+01, 1.200000e+01, 1.400000e+01]> : tensor<8xf32>
    %cst_0 = arith.constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00]> : tensor<8xf32>
    %0 = util.optimization_barrier %cst_0 : tensor<8xf32>
    %1 = util.optimization_barrier %cst_0 : tensor<8xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%0, %1 : tensor<8xf32>, tensor<8xf32>) outs(%0 : tensor<8xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %4 = arith.addf %in, %in_1 : f32
      linalg.yield %4 : f32
    } -> tensor<8xf32>
    %3 = util.optimization_barrier %cst : tensor<8xf32>
    "check.expect_almost_eq"(%2, %3) {rtol = 9.99999997E-7 : f32} : (tensor<8xf32>, tensor<8xf32>) -> ()
    return
  }
}
