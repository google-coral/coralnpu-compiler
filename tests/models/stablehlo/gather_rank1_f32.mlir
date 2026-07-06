func.func @main(%operand: tensor<?xf32>, %start_indices: tensor<?xi32>) -> tensor<?xf32> {
  %0 = "stablehlo.gather"(%operand, %start_indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [],
      collapsed_slice_dims = [0],
      start_index_map = [0],
      index_vector_dim = 1
    >,
    slice_sizes = array<i64: 1>
  } : (tensor<?xf32>, tensor<?xi32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}
