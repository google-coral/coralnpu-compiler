func.func @main(%arg0: tensor<?x?xi1>, %arg1: tensor<?x?xi16>, %arg2: tensor<?x?xi16>) -> tensor<?x?xi16> {
  %c0_idx = arith.constant 0 : index
  %c1_idx = arith.constant 1 : index
  %d0 = tensor.dim %arg1, %c0_idx : tensor<?x?xi16>
  %d1 = tensor.dim %arg1, %c1_idx : tensor<?x?xi16>
  %empty = tensor.empty(%d0, %d1) : tensor<?x?xi16>
  %0 = linalg.select ins(%arg0, %arg1, %arg2 : tensor<?x?xi1>, tensor<?x?xi16>, tensor<?x?xi16>) outs(%empty : tensor<?x?xi16>) -> tensor<?x?xi16>
  return %0 : tensor<?x?xi16>
}
