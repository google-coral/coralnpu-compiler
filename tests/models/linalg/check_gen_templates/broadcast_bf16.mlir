func.func @main(%arg0: tensor<?xbf16>, %arg1: tensor<?x?xbf16>) -> tensor<?x?xbf16> {
  %c0_idx = arith.constant 0 : index
  %c1_idx = arith.constant 1 : index
  %m = tensor.dim %arg1, %c0_idx : tensor<?x?xbf16>
  %n = tensor.dim %arg1, %c1_idx : tensor<?x?xbf16>

  %empty = tensor.empty(%m, %n) : tensor<?x?xbf16>
  %0 = linalg.broadcast ins(%arg0 : tensor<?xbf16>) outs(%empty : tensor<?x?xbf16>) dimensions = [0]
  return %0 : tensor<?x?xbf16>
}
