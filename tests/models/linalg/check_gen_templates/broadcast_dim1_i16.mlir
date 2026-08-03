// RUN: %template_path
func.func @main(%arg0: tensor<?xi16>, %arg1: tensor<?x?xi16>) -> tensor<?x?xi16> {
  %c0_idx = arith.constant 0 : index
  %c1_idx = arith.constant 1 : index
  %m = tensor.dim %arg1, %c0_idx : tensor<?x?xi16>
  %n = tensor.dim %arg1, %c1_idx : tensor<?x?xi16>

  %empty = tensor.empty(%m, %n) : tensor<?x?xi16>
  %0 = linalg.broadcast ins(%arg0 : tensor<?xi16>) outs(%empty : tensor<?x?xi16>) dimensions = [1]
  return %0 : tensor<?x?xi16>
}
