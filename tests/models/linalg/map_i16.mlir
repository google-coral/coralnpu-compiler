func.func @main(%arg0: tensor<?x?xi16>, %arg1: tensor<?x?xi16>) -> tensor<?x?xi16> {
  %c0_idx = arith.constant 0 : index
  %c1_idx = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0_idx : tensor<?x?xi16>
  %d1 = tensor.dim %arg0, %c1_idx : tensor<?x?xi16>

  %empty = tensor.empty(%d0, %d1) : tensor<?x?xi16>
  %0 = linalg.map ins(%arg0, %arg1 : tensor<?x?xi16>, tensor<?x?xi16>) outs(%empty : tensor<?x?xi16>) (%val0: i16, %val1: i16, %out: i16) {
    %res = arith.addi %val0, %val1 : i16
    linalg.yield %res : i16
  }
  return %0 : tensor<?x?xi16>
}
