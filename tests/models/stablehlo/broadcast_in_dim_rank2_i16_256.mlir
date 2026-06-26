func.func @main(%arg0: tensor<256xi16>) -> tensor<4x256xi16> {
  %0 = "stablehlo.broadcast_in_dim"(%arg0) {
    broadcast_dimensions = array<i64: 1>
  } : (tensor<256xi16>) -> tensor<4x256xi16>
  return %0 : tensor<4x256xi16>
}
