func.func @main(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0.0 : f32
  %c0_idx = arith.constant 0 : index
  %c2_idx = arith.constant 2 : index
  %batch = tensor.dim %arg1, %c0_idx : tensor<?x?x?xf32>
  %n = tensor.dim %arg1, %c2_idx : tensor<?x?x?xf32>
  %empty = tensor.empty(%batch, %n) : tensor<?x?xf32>
  %fill = linalg.fill ins(%c0 : f32) outs(%empty : tensor<?x?xf32>) -> tensor<?x?xf32>
  %0 = linalg.batch_vecmat
       ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?x?xf32>)
       outs(%fill : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
