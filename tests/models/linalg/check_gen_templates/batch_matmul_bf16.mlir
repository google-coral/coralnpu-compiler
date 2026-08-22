func.func @main(%arg0: tensor<?x?x?xbf16>, %arg1: tensor<?x?x?xbf16>) -> tensor<?x?x?xbf16> {
  %c0 = arith.constant 0.0 : bf16
  %c0_idx = arith.constant 0 : index
  %c1_idx = arith.constant 1 : index
  %c2_idx = arith.constant 2 : index
  %b = tensor.dim %arg0, %c0_idx : tensor<?x?x?xbf16>
  %m = tensor.dim %arg0, %c1_idx : tensor<?x?x?xbf16>
  %n = tensor.dim %arg1, %c2_idx : tensor<?x?x?xbf16>
  %empty = tensor.empty(%b, %m, %n) : tensor<?x?x?xbf16>
  %fill = linalg.fill ins(%c0 : bf16) outs(%empty : tensor<?x?x?xbf16>) -> tensor<?x?x?xbf16>
  %0 = linalg.batch_matmul ins(%arg0, %arg1 : tensor<?x?x?xbf16>, tensor<?x?x?xbf16>) outs(%fill : tensor<?x?x?xbf16>) -> tensor<?x?x?xbf16>
  return %0 : tensor<?x?x?xbf16>
}
