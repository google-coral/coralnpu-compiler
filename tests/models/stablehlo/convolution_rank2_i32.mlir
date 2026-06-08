func.func @main(%input: tensor<?x?xi32>, %kernel: tensor<?x?xi32>) -> tensor<?x?xi32> {
  %0 = "stablehlo.convolution"(%input, %kernel) {
    batch_group_count = 1 : i64,
    dimension_numbers = #stablehlo.conv<[b, f]x[i, o]->[b, f]>,
    feature_group_count = 1 : i64
  } : (tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
  return %0 : tensor<?x?xi32>
}
