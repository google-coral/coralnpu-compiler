func.func @main(%arg0: tensor<?x?xi32>, %arg1: tensor<?x?xi32>) -> tensor<?x?xi32> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<?x?xi32>
  return %0 : tensor<?x?xi32>
}
