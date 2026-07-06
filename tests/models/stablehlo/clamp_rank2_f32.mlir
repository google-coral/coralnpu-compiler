func.func @main(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %min = stablehlo.constant dense<-1.0> : tensor<f32>
  %max = stablehlo.constant dense<1.0> : tensor<f32>
  %0 = stablehlo.clamp %min, %arg0, %max : (tensor<f32>, tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
