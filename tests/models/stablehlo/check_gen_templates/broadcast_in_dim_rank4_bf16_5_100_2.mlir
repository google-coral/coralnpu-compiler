func.func @main(%arg0: tensor<5x100x2xbf16>) -> tensor<5x100x6x2xbf16> {
  %0 = "stablehlo.broadcast_in_dim"(%arg0) {
    broadcast_dimensions = array<i64: 0, 1, 3>
  } : (tensor<5x100x2xbf16>) -> tensor<5x100x6x2xbf16>
  return %0 : tensor<5x100x6x2xbf16>
}
