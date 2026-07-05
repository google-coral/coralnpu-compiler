// RUN: %template_path
func.func @main(%arg0: tensor<?x?xi16>) -> tensor<?xi16> {
  %c0 = arith.constant 0 : i16
  %c1_idx = arith.constant 1 : index
  %n = tensor.dim %arg0, %c1_idx : tensor<?x?xi16>
  %empty = tensor.empty(%n) : tensor<?xi16>
  %fill = linalg.fill ins(%c0 : i16) outs(%empty : tensor<?xi16>) -> tensor<?xi16>
  %0 = linalg.reduce ins(%arg0 : tensor<?x?xi16>) outs(%fill : tensor<?xi16>) dimensions = [0] (%in: i16, %out: i16) {
    %1 = arith.addi %in, %out : i16
    linalg.yield %1 : i16
  }
  return %0 : tensor<?xi16>
}
