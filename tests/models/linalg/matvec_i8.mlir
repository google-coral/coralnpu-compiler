func.func @main(%arg0: tensor<?x?xi8>, %arg1: tensor<?xi8>) -> tensor<?xi8> {
  %c0 = arith.constant 0 : i8
  %c0_idx = arith.constant 0 : index
  %m = tensor.dim %arg0, %c0_idx : tensor<?x?xi8>
  %empty = tensor.empty(%m) : tensor<?xi8>
  %fill = linalg.fill ins(%c0 : i8) outs(%empty : tensor<?xi8>) -> tensor<?xi8>
  %0 = linalg.matvec
       ins(%arg0, %arg1 : tensor<?x?xi8>, tensor<?xi8>)
       outs(%fill : tensor<?xi8>) -> tensor<?xi8>
  return %0 : tensor<?xi8>
}
