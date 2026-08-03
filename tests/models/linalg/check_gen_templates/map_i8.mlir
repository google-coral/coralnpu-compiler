func.func @main(%arg0: tensor<?x?xi8>, %arg1: tensor<?x?xi8>) -> tensor<?x?xi8> {
  %c0_idx = arith.constant 0 : index
  %c1_idx = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0_idx : tensor<?x?xi8>
  %d1 = tensor.dim %arg0, %c1_idx : tensor<?x?xi8>

  %empty = tensor.empty(%d0, %d1) : tensor<?x?xi8>
  %0 = linalg.map ins(%arg0, %arg1 : tensor<?x?xi8>, tensor<?x?xi8>) outs(%empty : tensor<?x?xi8>) (%val0: i8, %val1: i8, %out: i8) {
    %res = arith.addi %val0, %val1 : i8
    linalg.yield %res : i8
  }
  return %0 : tensor<?x?xi8>
}
