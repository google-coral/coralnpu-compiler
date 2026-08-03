#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  func.func @"generic_add_rank3_i8_2_3_4-2_3_4"() {
    %cst = arith.constant dense<[[[0, 2, 4, 6], [8, 10, 12, 14], [16, 18, 20, 22]], [[24, 26, 28, 30], [32, 34, 36, 38], [40, 42, 44, 46]]]> : tensor<2x3x4xi8>
    %cst_0 = arith.constant dense<[[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]> : tensor<2x3x4xi8>
    %0 = util.optimization_barrier %cst_0 : tensor<2x3x4xi8>
    %1 = util.optimization_barrier %cst_0 : tensor<2x3x4xi8>
    %2 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0, %1 : tensor<2x3x4xi8>, tensor<2x3x4xi8>) outs(%0 : tensor<2x3x4xi8>) {
    ^bb0(%in: i8, %in_1: i8, %out: i8):
      %4 = arith.addi %in, %in_1 : i8
      linalg.yield %4 : i8
    } -> tensor<2x3x4xi8>
    %3 = util.optimization_barrier %cst : tensor<2x3x4xi8>
    "check.expect_eq"(%2, %3) : (tensor<2x3x4xi8>, tensor<2x3x4xi8>) -> ()
    return
  }
}
