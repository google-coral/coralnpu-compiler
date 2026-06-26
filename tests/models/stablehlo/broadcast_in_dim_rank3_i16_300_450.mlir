func.func @main(%arg0: tensor<300x450xi16>) -> tensor<300x5x450xi16> {
  %0 = "stablehlo.broadcast_in_dim"(%arg0) {
    broadcast_dimensions = array<i64: 0, 2>
  } : (tensor<300x450xi16>) -> tensor<300x5x450xi16>
  return %0 : tensor<300x5x450xi16>
}
