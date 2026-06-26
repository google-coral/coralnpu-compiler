func.func @main(%arg0: tensor<2x3x4xi16>) -> tensor<2x3x6x4xi16> {
  %0 = "stablehlo.broadcast_in_dim"(%arg0) {
    broadcast_dimensions = array<i64: 0, 1, 3>
  } : (tensor<2x3x4xi16>) -> tensor<2x3x6x4xi16>
  return %0 : tensor<2x3x6x4xi16>
}
