// RUN: %template_path
func.func @main(%arg0: tensor<?x?xi16>) -> tensor<?xi16> {
  %min_val = arith.constant -32768 : i16
  %c0_idx = arith.constant 0 : index
  %m = tensor.dim %arg0, %c0_idx : tensor<?x?xi16>
  %empty = tensor.empty(%m) : tensor<?xi16>
  %fill = linalg.fill ins(%min_val : i16) outs(%empty : tensor<?xi16>) -> tensor<?xi16>
  %0 = linalg.reduce ins(%arg0 : tensor<?x?xi16>) outs(%fill : tensor<?xi16>) dimensions = [1] (%in: i16, %out: i16) {
    %1 = arith.maxsi %in, %out : i16
    linalg.yield %1 : i16
  }
  return %0 : tensor<?xi16>
}
