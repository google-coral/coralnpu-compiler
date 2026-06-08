func.func @main(%arg0: tensor<?xi32>, %arg1: tensor<?xi32>) -> tensor<?xi32> {
  %0 = stablehlo.multiply %arg0, %arg1 : tensor<?xi32>
  return %0 : tensor<?xi32>
}
