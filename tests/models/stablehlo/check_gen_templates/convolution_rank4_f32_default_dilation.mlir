func.func @main(%input: tensor<?x?x?x?xf32>, %kernel: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %0 = "stablehlo.convolution"(%input, %kernel) {
    batch_group_count = 1 : i64,
    dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>,
    feature_group_count = 1 : i64,
    padding = dense<0> : tensor<2x2xi64>,
    window_strides = array<i64: 1, 1>
  } : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}

