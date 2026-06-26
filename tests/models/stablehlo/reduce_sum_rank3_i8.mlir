func.func @main(%arg0: tensor<?x?x?xi8>) -> tensor<?x?xi8> {
  %c0 = stablehlo.constant dense<0> : tensor<i8>
  %0 = stablehlo.reduce(%arg0 init: %c0) applies stablehlo.add across dimensions = [2] : (tensor<?x?x?xi8>, tensor<i8>) -> tensor<?x?xi8>
  return %0 : tensor<?x?xi8>
}
