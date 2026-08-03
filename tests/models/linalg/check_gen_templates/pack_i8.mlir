func.func @main(%arg0: tensor<?x?xi8>) -> tensor<?x?x8x16xi8> {
  %c0_idx = arith.constant 0 : index
  %c1_idx = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0_idx : tensor<?x?xi8>
  %d1 = tensor.dim %arg0, %c1_idx : tensor<?x?xi8>

  %ts0 = arith.constant 8 : index
  %ts1 = arith.constant 16 : index

  %outer0 = arith.divsi %d0, %ts0 : index
  %outer1 = arith.divsi %d1, %ts1 : index

  %empty = tensor.empty(%outer0, %outer1) : tensor<?x?x8x16xi8>
  %0 = linalg.pack %arg0 inner_dims_pos = [0, 1] inner_tiles = [8, 16] into %empty : tensor<?x?xi8> -> tensor<?x?x8x16xi8>
  return %0 : tensor<?x?x8x16xi8>
}
