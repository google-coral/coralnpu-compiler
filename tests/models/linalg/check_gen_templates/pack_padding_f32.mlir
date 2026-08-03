// RUN: %template_path
func.func @main(%arg0: tensor<?x?xf32>) -> tensor<?x?x8x16xf32> {
  %pad_val = arith.constant 0.0 : f32
  %c0_idx = arith.constant 0 : index
  %c1_idx = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0_idx : tensor<?x?xf32>
  %d1 = tensor.dim %arg0, %c1_idx : tensor<?x?xf32>

  %ts0 = arith.constant 8 : index
  %ts1 = arith.constant 16 : index

  %c7 = arith.constant 7 : index
  %d0_padded = arith.addi %d0, %c7 : index
  %outer0 = arith.divsi %d0_padded, %ts0 : index

  %c15 = arith.constant 15 : index
  %d1_padded = arith.addi %d1, %c15 : index
  %outer1 = arith.divsi %d1_padded, %ts1 : index

  %empty = tensor.empty(%outer0, %outer1) : tensor<?x?x8x16xf32>
  %0 = linalg.pack %arg0 padding_value(%pad_val : f32) inner_dims_pos = [0, 1] inner_tiles = [8, 16] into %empty : tensor<?x?xf32> -> tensor<?x?x8x16xf32>
  return %0 : tensor<?x?x8x16xf32>
}
