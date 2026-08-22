// RUN: %template_path
func.func @main(%arg0: tensor<?x?x?xbf16>) -> tensor<?xbf16> {
  %c0 = arith.constant 0.0 : bf16
  %c0_idx = arith.constant 0 : index
  %l = tensor.dim %arg0, %c0_idx : tensor<?x?x?xbf16>
  %empty = tensor.empty(%l) : tensor<?xbf16>
  %fill = linalg.fill ins(%c0 : bf16) outs(%empty : tensor<?xbf16>) -> tensor<?xbf16>
  %0 = linalg.reduce ins(%arg0 : tensor<?x?x?xbf16>) outs(%fill : tensor<?xbf16>) dimensions = [1, 2] (%in: bf16, %out: bf16) {
    %1 = arith.addf %in, %out : bf16
    linalg.yield %1 : bf16
  }
  return %0 : tensor<?xbf16>
}
