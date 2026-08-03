func.func @main(%arg0: tensor<?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0_idx = arith.constant 0 : index
  %c1_idx = arith.constant 1 : index
  %m = tensor.dim %arg1, %c0_idx : tensor<?x?xf32>
  %n = tensor.dim %arg1, %c1_idx : tensor<?x?xf32>

  %empty = tensor.empty(%m, %n) : tensor<?x?xf32>
  %0 = linalg.broadcast ins(%arg0 : tensor<?xf32>) outs(%empty : tensor<?x?xf32>) dimensions = [0]
  return %0 : tensor<?x?xf32>
}
