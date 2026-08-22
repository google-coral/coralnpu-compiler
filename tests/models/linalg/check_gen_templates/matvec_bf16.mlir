func.func @main(%arg0: tensor<?x?xbf16>, %arg1: tensor<?xbf16>) -> tensor<?xbf16> {
  %c0 = arith.constant 0.0 : bf16
  %c0_idx = arith.constant 0 : index
  %m = tensor.dim %arg0, %c0_idx : tensor<?x?xbf16>
  %empty = tensor.empty(%m) : tensor<?xbf16>
  %fill = linalg.fill ins(%c0 : bf16) outs(%empty : tensor<?xbf16>) -> tensor<?xbf16>
  %0 = linalg.matvec
       ins(%arg0, %arg1 : tensor<?x?xbf16>, tensor<?xbf16>)
       outs(%fill : tensor<?xbf16>) -> tensor<?xbf16>
  return %0 : tensor<?xbf16>
}
