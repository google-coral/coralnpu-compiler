func.func @main(%arg0: tensor<?x?x?xi32>) -> tensor<?x?x?xi32> {
  %0 = stablehlo.abs %arg0 : tensor<?x?x?xi32>
  return %0 : tensor<?x?x?xi32>
}
