func.func @main(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %c0 = arith.constant 0.0 : f32
  %c0_idx = arith.constant 0 : index
  %c1_idx = arith.constant 1 : index
  %c2_idx = arith.constant 2 : index
  %b = tensor.dim %arg0, %c0_idx : tensor<?x?x?xf32>
  %m = tensor.dim %arg0, %c1_idx : tensor<?x?x?xf32>
  %n = tensor.dim %arg1, %c2_idx : tensor<?x?x?xf32>
  %empty = tensor.empty(%b, %m, %n) : tensor<?x?x?xf32>
  %fill = linalg.fill ins(%c0 : f32) outs(%empty : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %0 = linalg.batch_matmul ins(%arg0, %arg1 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%fill : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
}
