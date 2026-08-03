#map = affine_map<(d0) -> (d0)>
module {
  func.func @"generic_add_rank1_i32_8-8"() {
    %cst = arith.constant dense<[0, 2, 4, 6, 8, 10, 12, 14]> : tensor<8xi32>
    %cst_0 = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi32>
    %0 = util.optimization_barrier %cst_0 : tensor<8xi32>
    %1 = util.optimization_barrier %cst_0 : tensor<8xi32>
    %2 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%0, %1 : tensor<8xi32>, tensor<8xi32>) outs(%0 : tensor<8xi32>) {
    ^bb0(%in: i32, %in_1: i32, %out: i32):
      %4 = arith.addi %in, %in_1 : i32
      linalg.yield %4 : i32
    } -> tensor<8xi32>
    %3 = util.optimization_barrier %cst : tensor<8xi32>
    "check.expect_eq"(%2, %3) : (tensor<8xi32>, tensor<8xi32>) -> ()
    return
  }
}
