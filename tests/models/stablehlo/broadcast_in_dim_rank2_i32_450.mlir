func.func @main(%arg0: tensor<450xi32>) -> tensor<4x450xi32> {
  %0 = "stablehlo.broadcast_in_dim"(%arg0) {
    broadcast_dimensions = array<i64: 1>
  } : (tensor<450xi32>) -> tensor<4x450xi32>
  return %0 : tensor<4x450xi32>
}
