func.func @main(%arg0: tensor<?x?x?xbf16>) -> tensor<?x?xbf16> {
  %c0 = stablehlo.constant dense<0.0> : tensor<bf16>
  %0 = stablehlo.reduce(%arg0 init: %c0) applies stablehlo.add across dimensions = [2] : (tensor<?x?x?xbf16>, tensor<bf16>) -> tensor<?x?xbf16>
  return %0 : tensor<?x?xbf16>
}
