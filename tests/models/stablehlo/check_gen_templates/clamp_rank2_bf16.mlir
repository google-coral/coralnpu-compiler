func.func @main(%arg0: tensor<?x?xbf16>) -> tensor<?x?xbf16> {
  %min = stablehlo.constant dense<-1.0> : tensor<bf16>
  %max = stablehlo.constant dense<1.0> : tensor<bf16>
  %0 = stablehlo.clamp %min, %arg0, %max : (tensor<bf16>, tensor<?x?xbf16>, tensor<bf16>) -> tensor<?x?xbf16>
  return %0 : tensor<?x?xbf16>
}
