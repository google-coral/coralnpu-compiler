func.func @main(%arg0: tensor<?x?xi16>, %arg1: tensor<?x?xi16>) -> tensor<?x?xi1> {
  %0 = stablehlo.compare EQ, %arg0, %arg1, SIGNED : (tensor<?x?xi16>, tensor<?x?xi16>) -> tensor<?x?xi1>
  return %0 : tensor<?x?xi1>
}
