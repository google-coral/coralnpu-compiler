// RUN: %template_path
func.func @main(%arg0: tensor<?x?xi8>) -> tensor<?xi8> {
  %c0 = arith.constant 0 : i8
  %c1_idx = arith.constant 1 : index
  %n = tensor.dim %arg0, %c1_idx : tensor<?x?xi8>
  %empty = tensor.empty(%n) : tensor<?xi8>
  %fill = linalg.fill ins(%c0 : i8) outs(%empty : tensor<?xi8>) -> tensor<?xi8>
  %0 = linalg.reduce ins(%arg0 : tensor<?x?xi8>) outs(%fill : tensor<?xi8>) dimensions = [0] (%in: i8, %out: i8) {
    %1 = arith.addi %in, %out : i8
    linalg.yield %1 : i8
  }
  return %0 : tensor<?xi8>
}
