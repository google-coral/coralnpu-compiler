func.func @main(%arg0: tensor<?xf32>) -> tensor<4x?xf32> {
  %0 = stablehlo.broadcast %arg0, sizes = [4] : (tensor<?xf32>) -> tensor<4x?xf32>
  return %0 : tensor<4x?xf32>
}
