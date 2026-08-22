func.func @main(%arg0: tensor<?xbf16>) -> tensor<bf16> {
  %c0 = stablehlo.constant dense<0.0> : tensor<bf16>
  %0 = stablehlo.reduce(%arg0 init: %c0) applies stablehlo.add across dimensions = [0] : (tensor<?xbf16>, tensor<bf16>) -> tensor<bf16>
  return %0 : tensor<bf16>
}
