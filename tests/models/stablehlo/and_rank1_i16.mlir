func.func @main(%arg0: tensor<?xi16>, %arg1: tensor<?xi16>) -> tensor<?xi16> {
  %0 = stablehlo.and %arg0, %arg1 : (tensor<?xi16>, tensor<?xi16>) -> tensor<?xi16>
  return %0 : tensor<?xi16>
}
