func.func @main(%arg0: tensor<?xbf16>, %arg1: tensor<?x?xbf16>) -> tensor<?xbf16> {
  %c0 = arith.constant 0.0 : bf16
  %c1_idx = arith.constant 1 : index
  %n = tensor.dim %arg1, %c1_idx : tensor<?x?xbf16>
  %empty = tensor.empty(%n) : tensor<?xbf16>
  %fill = linalg.fill ins(%c0 : bf16) outs(%empty : tensor<?xbf16>) -> tensor<?xbf16>
  %0 = linalg.vecmat
       ins(%arg0, %arg1 : tensor<?xbf16>, tensor<?x?xbf16>)
       outs(%fill : tensor<?xbf16>) -> tensor<?xbf16>
  return %0 : tensor<?xbf16>
}
