func.func @main(%arg0: tensor<2x3x4xbf16>) -> tensor<2x3x6x4xbf16> {
  %0 = "stablehlo.broadcast_in_dim"(%arg0) {
    broadcast_dimensions = array<i64: 0, 1, 3>
  } : (tensor<2x3x4xbf16>) -> tensor<2x3x6x4xbf16>
  return %0 : tensor<2x3x6x4xbf16>
}
