module {
  func.func @"shift_left_rank2_i32_4_8-4_8"() {
    %cst = arith.constant dense<[[0, 2, 8, 24, 64, 160, 384, 896], [2048, 4608, 10240, 22528, 49152, 106496, 229376, 491520], [1048576, 2228224, 4718592, 9961472, 20971520, 44040192, 92274688, 192937984], [402653184, 838860800, 1744830464, -671088640, -1073741824, -1610612736, -2147483648, -2147483648]]> : tensor<4x8xi32>
    %cst_0 = arith.constant dense<[[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15], [16, 17, 18, 19, 20, 21, 22, 23], [24, 25, 26, 27, 28, 29, 30, 31]]> : tensor<4x8xi32>
    %0 = util.optimization_barrier %cst_0 : tensor<4x8xi32>
    %1 = util.optimization_barrier %cst_0 : tensor<4x8xi32>
    %2 = stablehlo.shift_left %0, %1 : tensor<4x8xi32>
    %3 = util.optimization_barrier %cst : tensor<4x8xi32>
    "check.expect_eq"(%2, %3) : (tensor<4x8xi32>, tensor<4x8xi32>) -> ()
    return
  }
}
