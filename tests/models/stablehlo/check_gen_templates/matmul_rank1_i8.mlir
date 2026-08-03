func.func @main(%arg0: tensor<?xi8>, %arg1: tensor<?xi8>) -> tensor<i8> {
  %0 = stablehlo.dot %arg0, %arg1 : (tensor<?xi8>, tensor<?xi8>) -> tensor<i8>
  return %0 : tensor<i8>
}
