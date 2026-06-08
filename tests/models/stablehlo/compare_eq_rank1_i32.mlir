func.func @main(%arg0: tensor<?xi32>, %arg1: tensor<?xi32>) -> tensor<?xi1> {
  %0 = stablehlo.compare EQ, %arg0, %arg1, SIGNED : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi1>
  return %0 : tensor<?xi1>
}
