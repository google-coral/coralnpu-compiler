// RUN: %template_path
func.func @main(%arg0: tensor<?xi32>, %arg1: tensor<?x?xi32>) -> tensor<?x?xi32> {
  %c0_idx = arith.constant 0 : index
  %c1_idx = arith.constant 1 : index
  %m = tensor.dim %arg1, %c0_idx : tensor<?x?xi32>
  %n = tensor.dim %arg1, %c1_idx : tensor<?x?xi32>

  %empty = tensor.empty(%m, %n) : tensor<?x?xi32>
  %0 = linalg.broadcast ins(%arg0 : tensor<?xi32>) outs(%empty : tensor<?x?xi32>) dimensions = [1]
  return %0 : tensor<?x?xi32>
}
