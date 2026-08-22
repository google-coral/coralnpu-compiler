func.func @main(%arg0: tensor<?xbf16>, %arg1: tensor<?xbf16>) -> tensor<bf16> {
  %c0 = arith.constant 0.0 : bf16
  %empty = tensor.empty() : tensor<bf16>
  %fill = linalg.fill ins(%c0 : bf16) outs(%empty : tensor<bf16>) -> tensor<bf16>
  %0 = linalg.dot
       ins(%arg0, %arg1 : tensor<?xbf16>, tensor<?xbf16>)
       outs(%fill : tensor<bf16>) -> tensor<bf16>
  return %0 : tensor<bf16>
}
