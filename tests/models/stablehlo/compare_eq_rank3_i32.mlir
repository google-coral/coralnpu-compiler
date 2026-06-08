func.func @main(%arg0: tensor<?x?x?xi32>, %arg1: tensor<?x?x?xi32>) -> tensor<?x?x?xi1> {
  %0 = stablehlo.compare EQ, %arg0, %arg1, SIGNED : (tensor<?x?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi1>
  return %0 : tensor<?x?x?xi1>
}
