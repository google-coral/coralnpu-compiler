module {
  func.func @reduce_sum_rank2_i32_4_8() {
    %cst = arith.constant dense<[28, 92, 156, 220]> : tensor<4xi32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %cst_0 = arith.constant dense<[[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15], [16, 17, 18, 19, 20, 21, 22, 23], [24, 25, 26, 27, 28, 29, 30, 31]]> : tensor<4x8xi32>
    %0 = util.optimization_barrier %cst_0 : tensor<4x8xi32>
    %1 = stablehlo.reduce(%0 init: %c) applies stablehlo.add across dimensions = [1] : (tensor<4x8xi32>, tensor<i32>) -> tensor<4xi32>
    %2 = util.optimization_barrier %cst : tensor<4xi32>
    "check.expect_eq"(%1, %2) : (tensor<4xi32>, tensor<4xi32>) -> ()
    return
  }
}
