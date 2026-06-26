func.func @main(%arg0: tensor<?xi8>, %arg1: tensor<?xi8>) -> tensor<?xi8> {
  %0 = stablehlo.divide %arg0, %arg1 : tensor<?xi8>
  return %0 : tensor<?xi8>
}
