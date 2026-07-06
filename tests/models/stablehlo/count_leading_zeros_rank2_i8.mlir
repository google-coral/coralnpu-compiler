func.func @main(%arg0: tensor<?x?xi8>) -> tensor<?x?xi8> {
  %0 = stablehlo.count_leading_zeros %arg0 : (tensor<?x?xi8>) -> tensor<?x?xi8>
  return %0 : tensor<?x?xi8>
}
