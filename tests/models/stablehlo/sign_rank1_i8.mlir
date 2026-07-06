func.func @main(%arg0: tensor<?xi8>) -> tensor<?xi8> {
  %0 = stablehlo.sign %arg0 : (tensor<?xi8>) -> tensor<?xi8>
  return %0 : tensor<?xi8>
}
