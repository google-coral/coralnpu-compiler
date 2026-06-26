func.func @main(%arg0: tensor<?xi16>) -> tensor<i16> {
  %c0 = stablehlo.constant dense<0> : tensor<i16>
  %0 = stablehlo.reduce(%arg0 init: %c0) applies stablehlo.add across dimensions = [0] : (tensor<?xi16>, tensor<i16>) -> tensor<i16>
  return %0 : tensor<i16>
}
