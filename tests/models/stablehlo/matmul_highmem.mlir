func.func @test_matmul_highmem() {
  %lhs = util.unfoldable_constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
  %rhs = util.unfoldable_constant dense<[[5, 6], [7, 8]]> : tensor<2x2xi32>
  %res = "stablehlo.dot"(%lhs, %rhs) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  %expected = util.unfoldable_constant dense<[[19, 22], [43, 50]]> : tensor<2x2xi32>
  check.expect_eq(%res, %expected) : tensor<2x2xi32>
  return
}
