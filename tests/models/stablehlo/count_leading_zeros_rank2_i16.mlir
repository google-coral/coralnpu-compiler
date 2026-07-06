func.func @main(%arg0: tensor<?x?xi16>) -> tensor<?x?xi16> {
  %0 = stablehlo.count_leading_zeros %arg0 : (tensor<?x?xi16>) -> tensor<?x?xi16>
  return %0 : tensor<?x?xi16>
}
