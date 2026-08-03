func.func @main(%arg0: tensor<8xi32>) -> tensor<4x8xi32> {
  %0 = "stablehlo.broadcast_in_dim"(%arg0) {
    broadcast_dimensions = array<i64: 1>
  } : (tensor<8xi32>) -> tensor<4x8xi32>
  return %0 : tensor<4x8xi32>
}
