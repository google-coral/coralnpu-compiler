func.func @main(%arg0: tensor<?x?x8x16xi16>) -> tensor<?x?xi16> {
  %c0_idx = arith.constant 0 : index
  %c1_idx = arith.constant 1 : index
  %outer0 = tensor.dim %arg0, %c0_idx : tensor<?x?x8x16xi16>
  %outer1 = tensor.dim %arg0, %c1_idx : tensor<?x?x8x16xi16>

  %ts0 = arith.constant 8 : index
  %ts1 = arith.constant 16 : index

  %d0 = arith.muli %outer0, %ts0 : index
  %d1 = arith.muli %outer1, %ts1 : index

  %empty = tensor.empty(%d0, %d1) : tensor<?x?xi16>
  %0 = linalg.unpack %arg0 inner_dims_pos = [0, 1] inner_tiles = [8, 16] into %empty : tensor<?x?x8x16xi16> -> tensor<?x?xi16>
  return %0 : tensor<?x?xi16>
}
