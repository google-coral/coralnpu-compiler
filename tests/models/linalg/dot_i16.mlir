func.func @main(%arg0: tensor<?xi16>, %arg1: tensor<?xi16>) -> tensor<i16> {
  %c0 = arith.constant 0 : i16
  %empty = tensor.empty() : tensor<i16>
  %fill = linalg.fill ins(%c0 : i16) outs(%empty : tensor<i16>) -> tensor<i16>
  %0 = linalg.dot
       ins(%arg0, %arg1 : tensor<?xi16>, tensor<?xi16>)
       outs(%fill : tensor<i16>) -> tensor<i16>
  return %0 : tensor<i16>
}
