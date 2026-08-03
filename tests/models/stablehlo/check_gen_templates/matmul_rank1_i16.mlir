func.func @main(%arg0: tensor<?xi16>, %arg1: tensor<?xi16>) -> tensor<i16> {
  %0 = stablehlo.dot %arg0, %arg1 : (tensor<?xi16>, tensor<?xi16>) -> tensor<i16>
  return %0 : tensor<i16>
}
