func.func @main(%arg0: tensor<?xi16>) -> tensor<?xi16> {
  %0 = stablehlo.count_leading_zeros %arg0 : (tensor<?xi16>) -> tensor<?xi16>
  return %0 : tensor<?xi16>
}
