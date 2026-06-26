func.func @main(%arg0: tensor<?x?xi16>) -> tensor<?x?xi16> {
  %0 = stablehlo.abs %arg0 : tensor<?x?xi16>
  return %0 : tensor<?x?xi16>
}
