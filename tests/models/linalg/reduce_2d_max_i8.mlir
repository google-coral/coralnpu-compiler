// RUN: %template_path
func.func @main(%arg0: tensor<?x?xi8>) -> tensor<?xi8> {
  %min_val = arith.constant -128 : i8
  %c0_idx = arith.constant 0 : index
  %m = tensor.dim %arg0, %c0_idx : tensor<?x?xi8>
  %empty = tensor.empty(%m) : tensor<?xi8>
  %fill = linalg.fill ins(%min_val : i8) outs(%empty : tensor<?xi8>) -> tensor<?xi8>
  %0 = linalg.reduce ins(%arg0 : tensor<?x?xi8>) outs(%fill : tensor<?xi8>) dimensions = [1] (%in: i8, %out: i8) {
    %1 = arith.maxsi %in, %out : i8
    linalg.yield %1 : i8
  }
  return %0 : tensor<?xi8>
}
