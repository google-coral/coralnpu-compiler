// RUN: %template_path
func.func @main(%arg0: tensor<?x?x?xi32>) -> tensor<?xi32> {
  %c0 = arith.constant 0 : i32
  %c0_idx = arith.constant 0 : index
  %l = tensor.dim %arg0, %c0_idx : tensor<?x?x?xi32>
  %empty = tensor.empty(%l) : tensor<?xi32>
  %fill = linalg.fill ins(%c0 : i32) outs(%empty : tensor<?xi32>) -> tensor<?xi32>
  %0 = linalg.reduce ins(%arg0 : tensor<?x?x?xi32>) outs(%fill : tensor<?xi32>) dimensions = [1, 2] (%in: i32, %out: i32) {
    %1 = arith.addi %in, %out : i32
    linalg.yield %1 : i32
  }
  return %0 : tensor<?xi32>
}
