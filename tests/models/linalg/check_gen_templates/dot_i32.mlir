func.func @main(%arg0: tensor<?xi32>, %arg1: tensor<?xi32>) -> tensor<i32> {
  %c0 = arith.constant 0 : i32
  %empty = tensor.empty() : tensor<i32>
  %fill = linalg.fill ins(%c0 : i32) outs(%empty : tensor<i32>) -> tensor<i32>
  %0 = linalg.dot
       ins(%arg0, %arg1 : tensor<?xi32>, tensor<?xi32>)
       outs(%fill : tensor<i32>) -> tensor<i32>
  return %0 : tensor<i32>
}
