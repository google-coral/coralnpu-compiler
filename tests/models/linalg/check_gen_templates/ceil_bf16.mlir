func.func @main(%arg0: tensor<?x?xbf16>) -> tensor<?x?xbf16> {
  %c0_idx = arith.constant 0 : index
  %c1_idx = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0_idx : tensor<?x?xbf16>
  %d1 = tensor.dim %arg0, %c1_idx : tensor<?x?xbf16>
  %empty = tensor.empty(%d0, %d1) : tensor<?x?xbf16>
  %0 = linalg.ceil ins(%arg0 : tensor<?x?xbf16>) outs(%empty : tensor<?x?xbf16>) -> tensor<?x?xbf16>
  return %0 : tensor<?x?xbf16>
}
