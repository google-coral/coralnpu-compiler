func.func @main(%arg0: tensor<?x?xi32>) -> tensor<?x?xi32> {
  %0 = stablehlo.add %arg0, %arg0 : tensor<?x?xi32>
  %1 = stablehlo.multiply %0, %arg0 : tensor<?x?xi32>
  return %1 : tensor<?x?xi32>
}
