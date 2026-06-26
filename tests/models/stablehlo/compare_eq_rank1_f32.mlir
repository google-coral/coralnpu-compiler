func.func @main(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xi1> {
  %0 = stablehlo.compare  EQ, %arg0, %arg1 : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xi1>
  return %0 : tensor<?xi1>
}
