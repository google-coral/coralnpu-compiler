func.func @main(%arg0: tensor<5x100x2xf32>) -> tensor<5x100x6x2xf32> {
  %0 = "stablehlo.broadcast_in_dim"(%arg0) {
    broadcast_dimensions = array<i64: 0, 1, 3>
  } : (tensor<5x100x2xf32>) -> tensor<5x100x6x2xf32>
  return %0 : tensor<5x100x6x2xf32>
}
