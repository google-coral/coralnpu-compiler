func.func @main(%input: tensor<?x?xf32>, %kernel: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = "stablehlo.convolution"(%input, %kernel) {
    batch_group_count = 1 : i64,
    dimension_numbers = #stablehlo.conv<[b, f]x[i, o]->[b, f]>,
    feature_group_count = 1 : i64
  } : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
