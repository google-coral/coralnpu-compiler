// RUN: %template_path
func.func @main(%arg0: tensor<?x?xi32>) -> tensor<?xi32> {
  %min_val = arith.constant -2147483648 : i32
  %c0_idx = arith.constant 0 : index
  %m = tensor.dim %arg0, %c0_idx : tensor<?x?xi32>
  %empty = tensor.empty(%m) : tensor<?xi32>
  %fill = linalg.fill ins(%min_val : i32) outs(%empty : tensor<?xi32>) -> tensor<?xi32>
  %0 = linalg.reduce ins(%arg0 : tensor<?x?xi32>) outs(%fill : tensor<?xi32>) dimensions = [1] (%in: i32, %out: i32) {
    %1 = arith.maxsi %in, %out : i32
    linalg.yield %1 : i32
  }
  return %0 : tensor<?xi32>
}
