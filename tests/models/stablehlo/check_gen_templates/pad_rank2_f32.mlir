func.func @main(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %cst = stablehlo.constant dense<0.0> : tensor<f32>
  %0 = stablehlo.pad %arg0, %cst, low = [1, 2], high = [1, 2], interior = [0, 0] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
