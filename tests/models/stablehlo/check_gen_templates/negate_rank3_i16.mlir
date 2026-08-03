func.func @main(%arg0: tensor<?x?x?xi16>) -> tensor<?x?x?xi16> {
  %0 = stablehlo.negate %arg0 : tensor<?x?x?xi16>
  return %0 : tensor<?x?x?xi16>
}
