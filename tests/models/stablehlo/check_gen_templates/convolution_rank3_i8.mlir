func.func @main(%input: tensor<?x?x?xi8>, %kernel: tensor<?x?x?xi8>) -> tensor<?x?x?xi8> {
  %0 = "stablehlo.convolution"(%input, %kernel) {
    batch_group_count = 1 : i64,
    dimension_numbers = #stablehlo.conv<[b, 0, f]x[0, i, o]->[b, 0, f]>,
    feature_group_count = 1 : i64,
    lhs_dilation = array<i64: 1>,
    padding = dense<0> : tensor<1x2xi64>,
    rhs_dilation = array<i64: 1>,
    window_strides = array<i64: 1>
  } : (tensor<?x?x?xi8>, tensor<?x?x?xi8>) -> tensor<?x?x?xi8>
  return %0 : tensor<?x?x?xi8>
}
