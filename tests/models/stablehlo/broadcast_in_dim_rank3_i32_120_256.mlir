func.func @main(%arg0: tensor<120x256xi32>) -> tensor<120x5x256xi32> {
  %0 = "stablehlo.broadcast_in_dim"(%arg0) {
    broadcast_dimensions = array<i64: 0, 2>
  } : (tensor<120x256xi32>) -> tensor<120x5x256xi32>
  return %0 : tensor<120x5x256xi32>
}
