func.func @main(%arg0: tensor<?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?xf32> {
  %c0 = arith.constant 0.0 : f32
  %c1_idx = arith.constant 1 : index
  %n = tensor.dim %arg1, %c1_idx : tensor<?x?xf32>
  %empty = tensor.empty(%n) : tensor<?xf32>
  %fill = linalg.fill ins(%c0 : f32) outs(%empty : tensor<?xf32>) -> tensor<?xf32>
  %0 = linalg.vecmat
       ins(%arg0, %arg1 : tensor<?xf32>, tensor<?x?xf32>)
       outs(%fill : tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}
