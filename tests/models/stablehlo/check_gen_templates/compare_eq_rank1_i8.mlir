func.func @main(%arg0: tensor<?xi8>, %arg1: tensor<?xi8>) -> tensor<?xi1> {
  %0 = stablehlo.compare EQ, %arg0, %arg1, SIGNED : (tensor<?xi8>, tensor<?xi8>) -> tensor<?xi1>
  return %0 : tensor<?xi1>
}
