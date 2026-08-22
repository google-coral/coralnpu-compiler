func.func @main(%arg0: tensor<?xbf16>, %arg1: tensor<?xbf16>) -> tensor<?xi1> {
  %0 = stablehlo.compare  EQ, %arg0, %arg1 : (tensor<?xbf16>, tensor<?xbf16>) -> tensor<?xi1>
  return %0 : tensor<?xi1>
}
