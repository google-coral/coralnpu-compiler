func.func @main(%arg0: tensor<?x?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
  %c0 = arith.constant 0.0 : f32
  %c0_idx = arith.constant 0 : index
  %m = tensor.dim %arg0, %c0_idx : tensor<?x?xf32>
  %empty = tensor.empty(%m) : tensor<?xf32>
  %fill = linalg.fill ins(%c0 : f32) outs(%empty : tensor<?xf32>) -> tensor<?xf32>
  %0 = linalg.matvec
       ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?xf32>)
       outs(%fill : tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}
