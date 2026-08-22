func.func @main(%arg0: tensor<450xbf16>) -> tensor<4x450xbf16> {
  %0 = "stablehlo.broadcast_in_dim"(%arg0) {
    broadcast_dimensions = array<i64: 1>
  } : (tensor<450xbf16>) -> tensor<4x450xbf16>
  return %0 : tensor<4x450xbf16>
}
