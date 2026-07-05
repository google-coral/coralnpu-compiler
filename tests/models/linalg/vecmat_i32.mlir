func.func @main(%arg0: tensor<?xi32>, %arg1: tensor<?x?xi32>) -> tensor<?xi32> {
  %c0 = arith.constant 0 : i32
  %c1_idx = arith.constant 1 : index
  %n = tensor.dim %arg1, %c1_idx : tensor<?x?xi32>
  %empty = tensor.empty(%n) : tensor<?xi32>
  %fill = linalg.fill ins(%c0 : i32) outs(%empty : tensor<?xi32>) -> tensor<?xi32>
  %0 = linalg.vecmat
       ins(%arg0, %arg1 : tensor<?xi32>, tensor<?x?xi32>)
       outs(%fill : tensor<?xi32>) -> tensor<?xi32>
  return %0 : tensor<?xi32>
}
