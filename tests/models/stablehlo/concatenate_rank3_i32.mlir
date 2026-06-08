func.func @main(%arg0: tensor<?x?x?xi32>, %arg1: tensor<?x?x?xi32>) -> tensor<?x?x?xi32> {
  %0 = stablehlo.concatenate %arg0, %arg1, dim = 2 : (tensor<?x?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  return %0 : tensor<?x?x?xi32>
}
