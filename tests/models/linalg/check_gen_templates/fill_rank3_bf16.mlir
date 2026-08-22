func.func @main(%arg0: tensor<?x?x?xbf16>) -> tensor<?x?x?xbf16> {
  %cst = arith.constant 1.0 : bf16
  %0 = linalg.fill ins(%cst : bf16) outs(%arg0 : tensor<?x?x?xbf16>) -> tensor<?x?x?xbf16>
  return %0 : tensor<?x?x?xbf16>
}
