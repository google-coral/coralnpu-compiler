func.func @main(%arg0: tensor<?x?xi8>, %arg1: tensor<?x?x?xi8>) -> tensor<?x?xi8> {
  %c0 = arith.constant 0 : i8
  %c0_idx = arith.constant 0 : index
  %c2_idx = arith.constant 2 : index
  %batch = tensor.dim %arg1, %c0_idx : tensor<?x?x?xi8>
  %n = tensor.dim %arg1, %c2_idx : tensor<?x?x?xi8>
  %empty = tensor.empty(%batch, %n) : tensor<?x?xi8>
  %fill = linalg.fill ins(%c0 : i8) outs(%empty : tensor<?x?xi8>) -> tensor<?x?xi8>
  %0 = linalg.batch_vecmat
       ins(%arg0, %arg1 : tensor<?x?xi8>, tensor<?x?x?xi8>)
       outs(%fill : tensor<?x?xi8>) -> tensor<?x?xi8>
  return %0 : tensor<?x?xi8>
}
