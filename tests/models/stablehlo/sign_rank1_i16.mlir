func.func @main(%arg0: tensor<?xi16>) -> tensor<?xi16> {
  %0 = stablehlo.sign %arg0 : (tensor<?xi16>) -> tensor<?xi16>
  return %0 : tensor<?xi16>
}
