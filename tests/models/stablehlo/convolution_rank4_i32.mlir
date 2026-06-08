func.func @main(%input: tensor<?x?x?x?xi32>, %kernel: tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32> {
  %0 = "stablehlo.convolution"(%input, %kernel) {
    batch_group_count = 1 : i64,
    dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>,
    feature_group_count = 1 : i64,
    lhs_dilation = array<i64: 1, 1>,
    padding = dense<0> : tensor<2x2xi64>,
    rhs_dilation = array<i64: 1, 1>,
    window_strides = array<i64: 1, 1>
  } : (tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  return %0 : tensor<?x?x?x?xi32>
}
