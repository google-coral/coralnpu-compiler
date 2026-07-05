func.func @main(%arg0: tensor<?x?x?xi8>, %arg1: tensor<?x?x?xi8>) -> tensor<?x?xi8> {
  %c0 = arith.constant 0 : i8
  %c0_idx = arith.constant 0 : index
  %c1_idx = arith.constant 1 : index
  %c2_idx = arith.constant 2 : index

  %m = tensor.dim %arg0, %c1_idx : tensor<?x?x?xi8>
  %n = tensor.dim %arg1, %c2_idx : tensor<?x?x?xi8>

  %empty = tensor.empty(%m, %n) : tensor<?x?xi8>
  %fill = linalg.fill ins(%c0 : i8) outs(%empty : tensor<?x?xi8>) -> tensor<?x?xi8>
  %0 = linalg.batch_reduce_matmul
       ins(%arg0, %arg1 : tensor<?x?x?xi8>, tensor<?x?x?xi8>)
       outs(%fill : tensor<?x?xi8>) -> tensor<?x?xi8>
  return %0 : tensor<?x?xi8>
}
