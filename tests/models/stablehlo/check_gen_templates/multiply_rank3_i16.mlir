func.func @main(%arg0: tensor<?x?x?xi16>, %arg1: tensor<?x?x?xi16>) -> tensor<?x?x?xi16> {
  %0 = stablehlo.multiply %arg0, %arg1 : tensor<?x?x?xi16>
  return %0 : tensor<?x?x?xi16>
}
