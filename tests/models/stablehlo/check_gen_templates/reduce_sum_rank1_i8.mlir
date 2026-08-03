func.func @main(%arg0: tensor<?xi8>) -> tensor<i8> {
  %c0 = stablehlo.constant dense<0> : tensor<i8>
  %0 = stablehlo.reduce(%arg0 init: %c0) applies stablehlo.add across dimensions = [0] : (tensor<?xi8>, tensor<i8>) -> tensor<i8>
  return %0 : tensor<i8>
}
