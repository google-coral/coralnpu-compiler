func.func @test_mini_multi_dispatch() {
  %input = util.unfoldable_constant dense<[[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16], [17, 18, 19, 20, 21, 22, 23, 24], [25, 26, 27, 28, 29, 30, 31, 32]]> : tensor<4x8xi32>
  %w1 = stablehlo.constant dense<[[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8], [6, 7, 8, 9], [7, 8, 9, 10], [8, 9, 10, 11]]> : tensor<8x4xi32>
  %h1 = stablehlo.dot %input, %w1 : (tensor<4x8xi32>, tensor<8x4xi32>) -> tensor<4x4xi32>
  %b1 = stablehlo.constant dense<10> : tensor<4x4xi32>
  %act1 = stablehlo.add %h1, %b1 : tensor<4x4xi32>

  %w2 = stablehlo.constant dense<[[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]]> : tensor<4x4xi32>
  %h2 = stablehlo.dot %act1, %w2 : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
  %b2 = stablehlo.constant dense<5> : tensor<4x4xi32>
  %res = stablehlo.add %h2, %b2 : tensor<4x4xi32>

  %expected = util.unfoldable_constant dense<[[505, 541, 541, 577], [1209, 1309, 1309, 1409], [1913, 2077, 2077, 2241], [2617, 2845, 2845, 3073]]> : tensor<4x4xi32>
  check.expect_eq(%res, %expected) : tensor<4x4xi32>
  return
}
