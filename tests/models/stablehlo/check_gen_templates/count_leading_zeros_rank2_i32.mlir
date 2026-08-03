func.func @main(%arg0: tensor<?x?xi32>) -> tensor<?x?xi32> {
  %0 = stablehlo.count_leading_zeros %arg0 : (tensor<?x?xi32>) -> tensor<?x?xi32>
  return %0 : tensor<?x?xi32>
}
