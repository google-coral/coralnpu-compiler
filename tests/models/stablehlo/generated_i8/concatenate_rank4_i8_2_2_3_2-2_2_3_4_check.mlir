module {
  func.func @"concatenate_rank4_i8_2_2_3_2-2_2_3_4"() {
    %cst = arith.constant dense<[[[[0, 1, 0, 1, 2, 3], [2, 3, 4, 5, 6, 7], [4, 5, 8, 9, 10, 11]], [[6, 7, 12, 13, 14, 15], [8, 9, 16, 17, 18, 19], [10, 11, 20, 21, 22, 23]]], [[[12, 13, 24, 25, 26, 27], [14, 15, 28, 29, 30, 31], [16, 17, 32, 33, 34, 35]], [[18, 19, 36, 37, 38, 39], [20, 21, 40, 41, 42, 43], [22, 23, 44, 45, 46, 47]]]]> : tensor<2x2x3x6xi8>
    %cst_0 = arith.constant dense<[[[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]], [[[24, 25, 26, 27], [28, 29, 30, 31], [32, 33, 34, 35]], [[36, 37, 38, 39], [40, 41, 42, 43], [44, 45, 46, 47]]]]> : tensor<2x2x3x4xi8>
    %cst_1 = arith.constant dense<[[[[0, 1], [2, 3], [4, 5]], [[6, 7], [8, 9], [10, 11]]], [[[12, 13], [14, 15], [16, 17]], [[18, 19], [20, 21], [22, 23]]]]> : tensor<2x2x3x2xi8>
    %0 = util.optimization_barrier %cst_1 : tensor<2x2x3x2xi8>
    %1 = util.optimization_barrier %cst_0 : tensor<2x2x3x4xi8>
    %2 = stablehlo.concatenate %0, %1, dim = 3 : (tensor<2x2x3x2xi8>, tensor<2x2x3x4xi8>) -> tensor<2x2x3x6xi8>
    %3 = util.optimization_barrier %cst : tensor<2x2x3x6xi8>
    "check.expect_eq"(%2, %3) : (tensor<2x2x3x6xi8>, tensor<2x2x3x6xi8>) -> ()
    return
  }
}
