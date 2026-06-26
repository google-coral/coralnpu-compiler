func.func @main(%arg0: tensor<4x8xi8>) -> tensor<4x5x8xi8> {
  %0 = "stablehlo.broadcast_in_dim"(%arg0) {
    broadcast_dimensions = array<i64: 0, 2>
  } : (tensor<4x8xi8>) -> tensor<4x5x8xi8>
  return %0 : tensor<4x5x8xi8>
}
