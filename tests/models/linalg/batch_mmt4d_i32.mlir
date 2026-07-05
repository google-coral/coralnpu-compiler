func.func @main(%arg0: tensor<?x?x?x?x?xi32>, %arg1: tensor<?x?x?x?x?xi32>) -> tensor<?x?x?x?x?xi32> {
  %c0 = arith.constant 0 : i32
  %c0_idx = arith.constant 0 : index
  %c1_idx = arith.constant 1 : index
  %c2_idx = arith.constant 2 : index
  %c3_idx = arith.constant 3 : index

  %batch = tensor.dim %arg0, %c0_idx : tensor<?x?x?x?x?xi32>
  %m1 = tensor.dim %arg0, %c1_idx : tensor<?x?x?x?x?xi32>
  %n1 = tensor.dim %arg1, %c1_idx : tensor<?x?x?x?x?xi32>
  %m0 = tensor.dim %arg0, %c3_idx : tensor<?x?x?x?x?xi32>
  %n0 = tensor.dim %arg1, %c3_idx : tensor<?x?x?x?x?xi32>

  %empty = tensor.empty(%batch, %m1, %n1, %m0, %n0) : tensor<?x?x?x?x?xi32>
  %fill = linalg.fill ins(%c0 : i32) outs(%empty : tensor<?x?x?x?x?xi32>) -> tensor<?x?x?x?x?xi32>
  %0 = linalg.batch_mmt4d
       ins(%arg0, %arg1 : tensor<?x?x?x?x?xi32>, tensor<?x?x?x?x?xi32>)
       outs(%fill : tensor<?x?x?x?x?xi32>) -> tensor<?x?x?x?x?xi32>
  return %0 : tensor<?x?x?x?x?xi32>
}
