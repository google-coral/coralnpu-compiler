func.func @main(%arg0: tensor<?xi8>, %arg1: tensor<?x?xi8>) -> tensor<?x?xi8> {
  %c0_idx = arith.constant 0 : index
  %c1_idx = arith.constant 1 : index
  %m = tensor.dim %arg1, %c0_idx : tensor<?x?xi8>
  %n = tensor.dim %arg1, %c1_idx : tensor<?x?xi8>

  %empty = tensor.empty(%m, %n) : tensor<?x?xi8>
  %0 = linalg.broadcast ins(%arg0 : tensor<?xi8>) outs(%empty : tensor<?x?xi8>) dimensions = [0]
  return %0 : tensor<?x?xi8>
}
