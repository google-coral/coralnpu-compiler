func.func @main(%arg0: tensor<450xf32>) -> tensor<4x450xf32> {
  %0 = "stablehlo.broadcast_in_dim"(%arg0) {
    broadcast_dimensions = array<i64: 1>
  } : (tensor<450xf32>) -> tensor<4x450xf32>
  return %0 : tensor<4x450xf32>
}
