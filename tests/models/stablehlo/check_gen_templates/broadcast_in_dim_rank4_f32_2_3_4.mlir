func.func @main(%arg0: tensor<2x3x4xf32>) -> tensor<2x3x6x4xf32> {
  %0 = "stablehlo.broadcast_in_dim"(%arg0) {
    broadcast_dimensions = array<i64: 0, 1, 3>
  } : (tensor<2x3x4xf32>) -> tensor<2x3x6x4xf32>
  return %0 : tensor<2x3x6x4xf32>
}
