func.func @main(%arg0: tensor<300x450xi32>) -> tensor<300x5x450xi32> {
  %0 = "stablehlo.broadcast_in_dim"(%arg0) {
    broadcast_dimensions = array<i64: 0, 2>
  } : (tensor<300x450xi32>) -> tensor<300x5x450xi32>
  return %0 : tensor<300x5x450xi32>
}
