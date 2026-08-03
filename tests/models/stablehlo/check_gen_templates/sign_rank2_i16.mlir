func.func @main(%arg0: tensor<?x?xi16>) -> tensor<?x?xi16> {
  %0 = stablehlo.sign %arg0 : (tensor<?x?xi16>) -> tensor<?x?xi16>
  return %0 : tensor<?x?xi16>
}
