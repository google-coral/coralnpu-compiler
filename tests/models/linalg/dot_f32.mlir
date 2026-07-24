func.func @main(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<f32> attributes {check.atol = 1.01e-4 : f32, check.rtol = 2.0e-6 : f32} {
  %c0 = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<f32>
  %fill = linalg.fill ins(%c0 : f32) outs(%empty : tensor<f32>) -> tensor<f32>
  %0 = linalg.dot
       ins(%arg0, %arg1 : tensor<?xf32>, tensor<?xf32>)
       outs(%fill : tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}
