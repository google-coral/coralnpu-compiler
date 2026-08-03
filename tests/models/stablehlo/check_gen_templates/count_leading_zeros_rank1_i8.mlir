func.func @main(%arg0: tensor<?xi8>) -> tensor<?xi8> {
  %0 = stablehlo.count_leading_zeros %arg0 : (tensor<?xi8>) -> tensor<?xi8>
  return %0 : tensor<?xi8>
}
