func.func @main(%arg0: tensor<?xf32>) -> tensor<f32> {
  %c0 = stablehlo.constant dense<0.0> : tensor<f32>
  %0 = stablehlo.reduce(%arg0 init: %c0) applies stablehlo.add across dimensions = [0] : (tensor<?xf32>, tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}
