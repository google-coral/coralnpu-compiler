module {
  func.func @"convolution_rank2_i32_4_8-8_4"() {
    %cst = arith.constant dense<[[560, 588, 616, 644], [1456, 1548, 1640, 1732], [2352, 2508, 2664, 2820], [3248, 3468, 3688, 3908]]> : tensor<4x4xi32>
    %cst_0 = arith.constant dense<[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23], [24, 25, 26, 27], [28, 29, 30, 31]]> : tensor<8x4xi32>
    %cst_1 = arith.constant dense<[[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15], [16, 17, 18, 19, 20, 21, 22, 23], [24, 25, 26, 27, 28, 29, 30, 31]]> : tensor<4x8xi32>
    %0 = util.optimization_barrier %cst_1 : tensor<4x8xi32>
    %1 = util.optimization_barrier %cst_0 : tensor<8x4xi32>
    %2 = stablehlo.convolution(%0, %1) dim_numbers = [b, f]x[i, o]->[b, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<4x8xi32>, tensor<8x4xi32>) -> tensor<4x4xi32>
    %3 = util.optimization_barrier %cst : tensor<4x4xi32>
    "check.expect_eq"(%2, %3) : (tensor<4x4xi32>, tensor<4x4xi32>) -> ()
    return
  }
}
