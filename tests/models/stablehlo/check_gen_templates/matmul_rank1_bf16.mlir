func.func @main(%arg0: tensor<?xbf16>, %arg1: tensor<?xbf16>) -> tensor<bf16> {
  %0 = stablehlo.dot %arg0, %arg1 : (tensor<?xbf16>, tensor<?xbf16>) -> tensor<bf16>
  return %0 : tensor<bf16>
}
