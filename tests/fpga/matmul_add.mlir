func.func @test_matmul_add() {
  %arg0 = util.unfoldable_constant dense<[[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16], [17, 18, 19, 20, 21, 22, 23, 24], [25, 26, 27, 28, 29, 30, 31, 32]]> : tensor<4x8xi32>
  %arg1 = util.unfoldable_constant dense<[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24], [25, 26, 27, 28], [29, 30, 31, 32]]> : tensor<8x4xi32>
  %arg2 = util.unfoldable_constant dense<[[10, 11, 12, 13], [14, 15, 16, 17], [18, 19, 20, 21], [22, 23, 24, 25]]> : tensor<4x4xi32>
  %0 = stablehlo.dot %arg0, %arg1 : (tensor<4x8xi32>, tensor<8x4xi32>) -> tensor<4x4xi32>
  %res = stablehlo.add %0, %arg2 : tensor<4x4xi32>
  %expected = util.unfoldable_constant dense<[[718, 755, 792, 829], [1682, 1783, 1884, 1985], [2646, 2811, 2976, 3141], [3610, 3839, 4068, 4297]]> : tensor<4x4xi32>
  check.expect_eq(%res, %expected) : tensor<4x4xi32>
  return
}
