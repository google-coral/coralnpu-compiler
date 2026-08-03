func.func @main(%arg0: tensor<?x?xi32>, %arg1: tensor<?x?x?xi32>) -> tensor<?x?xi32> {
  %c0 = arith.constant 0 : i32
  %c0_idx = arith.constant 0 : index
  %c2_idx = arith.constant 2 : index
  %batch = tensor.dim %arg1, %c0_idx : tensor<?x?x?xi32>
  %n = tensor.dim %arg1, %c2_idx : tensor<?x?x?xi32>
  %empty = tensor.empty(%batch, %n) : tensor<?x?xi32>
  %fill = linalg.fill ins(%c0 : i32) outs(%empty : tensor<?x?xi32>) -> tensor<?x?xi32>
  %0 = linalg.batch_vecmat
       ins(%arg0, %arg1 : tensor<?x?xi32>, tensor<?x?x?xi32>)
       outs(%fill : tensor<?x?xi32>) -> tensor<?x?xi32>
  return %0 : tensor<?x?xi32>
}
