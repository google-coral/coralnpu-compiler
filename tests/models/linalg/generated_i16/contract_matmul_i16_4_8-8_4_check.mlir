#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @"contract_matmul_i16_4_8-8_4"() {
    %cst = arith.constant dense<[[560, 588, 616, 644], [1456, 1548, 1640, 1732], [2352, 2508, 2664, 2820], [3248, 3468, 3688, 3908]]> : tensor<4x4xi16>
    %cst_0 = arith.constant dense<[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23], [24, 25, 26, 27], [28, 29, 30, 31]]> : tensor<8x4xi16>
    %c0_i16 = arith.constant 0 : i16
    %cst_1 = arith.constant dense<[[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15], [16, 17, 18, 19, 20, 21, 22, 23], [24, 25, 26, 27, 28, 29, 30, 31]]> : tensor<4x8xi16>
    %0 = util.optimization_barrier %cst_1 : tensor<4x8xi16>
    %1 = util.optimization_barrier %cst_0 : tensor<8x4xi16>
    %2 = tensor.empty() : tensor<4x4xi16>
    %3 = linalg.fill ins(%c0_i16 : i16) outs(%2 : tensor<4x4xi16>) -> tensor<4x4xi16>
    %4 = linalg.contract indexing_maps = [#map, #map1, #map2] ins(%0, %1 : tensor<4x8xi16>, tensor<8x4xi16>) outs(%3 : tensor<4x4xi16>) -> tensor<4x4xi16>
    %5 = util.optimization_barrier %cst : tensor<4x4xi16>
    "check.expect_eq"(%4, %5) : (tensor<4x4xi16>, tensor<4x4xi16>) -> ()
    return
  }
}
