func.func @main(%arg0: tensor<120x256xbf16>) -> tensor<120x5x256xbf16> {
  %0 = "stablehlo.broadcast_in_dim"(%arg0) {
    broadcast_dimensions = array<i64: 0, 2>
  } : (tensor<120x256xbf16>) -> tensor<120x5x256xbf16>
  return %0 : tensor<120x5x256xbf16>
}
