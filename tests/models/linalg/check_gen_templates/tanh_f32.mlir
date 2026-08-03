func.func @main(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0_idx = arith.constant 0 : index
  %c1_idx = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0_idx : tensor<?x?xf32>
  %d1 = tensor.dim %arg0, %c1_idx : tensor<?x?xf32>
  %empty = tensor.empty(%d0, %d1) : tensor<?x?xf32>
  %0 = linalg.tanh ins(%arg0 : tensor<?x?xf32>) outs(%empty : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
