func.func @main(%arg0: tensor<5x100x2xi32>) -> tensor<5x100x6x2xi32> {
  %0 = "stablehlo.broadcast_in_dim"(%arg0) {
    broadcast_dimensions = array<i64: 0, 1, 3>
  } : (tensor<5x100x2xi32>) -> tensor<5x100x6x2xi32>
  return %0 : tensor<5x100x6x2xi32>
}
