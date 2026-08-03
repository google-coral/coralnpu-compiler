#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
module {
  func.func @generic_reduction_2d_i16_4_8() {
    %cst = arith.constant dense<[28, 92, 156, 220]> : tensor<4xi16>
    %c0_i16 = arith.constant 0 : i16
    %cst_0 = arith.constant dense<[[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15], [16, 17, 18, 19, 20, 21, 22, 23], [24, 25, 26, 27, 28, 29, 30, 31]]> : tensor<4x8xi16>
    %0 = util.optimization_barrier %cst_0 : tensor<4x8xi16>
    %1 = tensor.empty() : tensor<4xi16>
    %2 = linalg.fill ins(%c0_i16 : i16) outs(%1 : tensor<4xi16>) -> tensor<4xi16>
    %3 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%0 : tensor<4x8xi16>) outs(%2 : tensor<4xi16>) {
    ^bb0(%in: i16, %out: i16):
      %5 = arith.addi %in, %out : i16
      linalg.yield %5 : i16
    } -> tensor<4xi16>
    %4 = util.optimization_barrier %cst : tensor<4xi16>
    "check.expect_eq"(%3, %4) : (tensor<4xi16>, tensor<4xi16>) -> ()
    return
  }
}
