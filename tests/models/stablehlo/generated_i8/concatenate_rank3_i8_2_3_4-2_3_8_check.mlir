module {
  func.func @"concatenate_rank3_i8_2_3_4-2_3_8"() {
    %cst = arith.constant dense<[[[0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7], [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], [8, 9, 10, 11, 16, 17, 18, 19, 20, 21, 22, 23]], [[12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31], [16, 17, 18, 19, 32, 33, 34, 35, 36, 37, 38, 39], [20, 21, 22, 23, 40, 41, 42, 43, 44, 45, 46, 47]]]> : tensor<2x3x12xi8>
    %cst_0 = arith.constant dense<[[[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15], [16, 17, 18, 19, 20, 21, 22, 23]], [[24, 25, 26, 27, 28, 29, 30, 31], [32, 33, 34, 35, 36, 37, 38, 39], [40, 41, 42, 43, 44, 45, 46, 47]]]> : tensor<2x3x8xi8>
    %cst_1 = arith.constant dense<[[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]> : tensor<2x3x4xi8>
    %0 = util.optimization_barrier %cst_1 : tensor<2x3x4xi8>
    %1 = util.optimization_barrier %cst_0 : tensor<2x3x8xi8>
    %2 = stablehlo.concatenate %0, %1, dim = 2 : (tensor<2x3x4xi8>, tensor<2x3x8xi8>) -> tensor<2x3x12xi8>
    %3 = util.optimization_barrier %cst : tensor<2x3x12xi8>
    "check.expect_eq"(%2, %3) : (tensor<2x3x12xi8>, tensor<2x3x12xi8>) -> ()
    return
  }
}
