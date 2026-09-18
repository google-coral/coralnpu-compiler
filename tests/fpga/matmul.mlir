func.func @test_matmul() {
  %arg0 = util.unfoldable_constant dense<[[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16], [17, 18, 19, 20, 21, 22, 23, 24], [25, 26, 27, 28, 29, 30, 31, 32]]> : tensor<4x8xi32>
  %arg1 = util.unfoldable_constant dense<[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24], [25, 26, 27, 28], [29, 30, 31, 32]]> : tensor<8x4xi32>
  %res = stablehlo.dot %arg0, %arg1 : (tensor<4x8xi32>, tensor<8x4xi32>) -> tensor<4x4xi32>
  %expected = util.unfoldable_constant dense<[[708, 744, 780, 816], [1668, 1768, 1868, 1968], [2628, 2792, 2956, 3120], [3588, 3816, 4044, 4272]]> : tensor<4x4xi32>
  check.expect_eq(%res, %expected) : tensor<4x4xi32>
  return
}
