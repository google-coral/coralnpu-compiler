// RUN: %template_path
func.func @main(%arg0: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %c0_idx = arith.constant 0 : index
  %c1_idx = arith.constant 1 : index
  %c2_idx = arith.constant 2 : index
  %d0 = tensor.dim %arg0, %c0_idx : tensor<?x?x?xf32>
  %d1 = tensor.dim %arg0, %c1_idx : tensor<?x?x?xf32>
  %d2 = tensor.dim %arg0, %c2_idx : tensor<?x?x?xf32>

  %empty = tensor.empty(%d0, %d2, %d1) : tensor<?x?x?xf32>
  %0 = linalg.transpose ins(%arg0 : tensor<?x?x?xf32>) outs(%empty : tensor<?x?x?xf32>) permutation = [0, 2, 1]
  return %0 : tensor<?x?x?xf32>
}
