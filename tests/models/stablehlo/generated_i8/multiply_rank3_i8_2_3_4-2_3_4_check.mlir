module {
  func.func @"multiply_rank3_i8_2_3_4-2_3_4"() {
    %cst = arith.constant dense<[[[0, 1, 4, 9], [16, 25, 36, 49], [64, 81, 100, 121]], [[-112, -87, -60, -31], [0, 33, 68, 105], [-112, -71, -28, 17]]]> : tensor<2x3x4xi8>
    %cst_0 = arith.constant dense<[[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]> : tensor<2x3x4xi8>
    %0 = util.optimization_barrier %cst_0 : tensor<2x3x4xi8>
    %1 = util.optimization_barrier %cst_0 : tensor<2x3x4xi8>
    %2 = stablehlo.multiply %0, %1 : tensor<2x3x4xi8>
    %3 = util.optimization_barrier %cst : tensor<2x3x4xi8>
    "check.expect_eq"(%2, %3) : (tensor<2x3x4xi8>, tensor<2x3x4xi8>) -> ()
    return
  }
}
