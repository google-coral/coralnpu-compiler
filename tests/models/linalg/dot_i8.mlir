func.func @main(%arg0: tensor<?xi8>, %arg1: tensor<?xi8>) -> tensor<i8> {
  %c0 = arith.constant 0 : i8
  %empty = tensor.empty() : tensor<i8>
  %fill = linalg.fill ins(%c0 : i8) outs(%empty : tensor<i8>) -> tensor<i8>
  %0 = linalg.dot
       ins(%arg0, %arg1 : tensor<?xi8>, tensor<?xi8>)
       outs(%fill : tensor<i8>) -> tensor<i8>
  return %0 : tensor<i8>
}
