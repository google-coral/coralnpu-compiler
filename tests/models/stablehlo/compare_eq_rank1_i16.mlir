func.func @main(%arg0: tensor<?xi16>, %arg1: tensor<?xi16>) -> tensor<?xi1> {
  %0 = stablehlo.compare EQ, %arg0, %arg1, SIGNED : (tensor<?xi16>, tensor<?xi16>) -> tensor<?xi1>
  return %0 : tensor<?xi1>
}
