func.func @main(%arg0: tensor<?x?xi16>, %arg1: tensor<?x?xi16>) -> tensor<?x?xi16> {
  %0 = stablehlo.divide %arg0, %arg1 : tensor<?x?xi16>
  return %0 : tensor<?x?xi16>
}
