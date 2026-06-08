func.func @main(%arg0: tensor<?xi32>, %arg1: tensor<?xi32>) -> tensor<i32> {
  %0 = stablehlo.dot %arg0, %arg1 : (tensor<?xi32>, tensor<?xi32>) -> tensor<i32>
  return %0 : tensor<i32>
}
