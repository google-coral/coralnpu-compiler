// RUN: %template_path
func.func @main(%arg0: tensor<?x?xbf16>) -> tensor<?xbf16> {
  %min_val = arith.constant -3.40282347E+38 : bf16
  %c0_idx = arith.constant 0 : index
  %m = tensor.dim %arg0, %c0_idx : tensor<?x?xbf16>
  %empty = tensor.empty(%m) : tensor<?xbf16>
  %fill = linalg.fill ins(%min_val : bf16) outs(%empty : tensor<?xbf16>) -> tensor<?xbf16>
  %0 = linalg.reduce ins(%arg0 : tensor<?x?xbf16>) outs(%fill : tensor<?xbf16>) dimensions = [1] (%in: bf16, %out: bf16) {
    %1 = arith.maximumf %in, %out : bf16
    linalg.yield %1 : bf16
  }
  return %0 : tensor<?xbf16>
}
