#map = affine_map<(d0) -> (d0)>
module {
  func.func @"generic_add_rank1_i8_8-8"() {
    %cst = arith.constant dense<[0, 2, 4, 6, 8, 10, 12, 14]> : tensor<8xi8>
    %cst_0 = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi8>
    %0 = util.optimization_barrier %cst_0 : tensor<8xi8>
    %1 = util.optimization_barrier %cst_0 : tensor<8xi8>
    %2 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%0, %1 : tensor<8xi8>, tensor<8xi8>) outs(%0 : tensor<8xi8>) {
    ^bb0(%in: i8, %in_1: i8, %out: i8):
      %4 = arith.addi %in, %in_1 : i8
      linalg.yield %4 : i8
    } -> tensor<8xi8>
    %3 = util.optimization_barrier %cst : tensor<8xi8>
    "check.expect_eq"(%2, %3) : (tensor<8xi8>, tensor<8xi8>) -> ()
    return
  }
}
