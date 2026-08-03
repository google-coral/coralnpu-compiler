func.func @main(%arg0: tensor<256xi8>) -> tensor<4x256xi8> {
  %0 = "stablehlo.broadcast_in_dim"(%arg0) {
    broadcast_dimensions = array<i64: 1>
  } : (tensor<256xi8>) -> tensor<4x256xi8>
  return %0 : tensor<4x256xi8>
}
