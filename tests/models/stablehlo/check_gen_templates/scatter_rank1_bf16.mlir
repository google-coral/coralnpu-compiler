func.func @main(%input: tensor<?xbf16>, %scatter_indices: tensor<?xi32>, %updates: tensor<?xbf16>) -> tensor<?xbf16> {
  %0 = "stablehlo.scatter"(%input, %scatter_indices, %updates) ({
  ^bb0(%lhs: tensor<bf16>, %rhs: tensor<bf16>):
     %res = stablehlo.add %lhs, %rhs : tensor<bf16>
     "stablehlo.return"(%res) : (tensor<bf16>) -> ()
  }) {
     indices_are_sorted = false,
     scatter_dimension_numbers = #stablehlo.scatter<
       update_window_dims = [],
       inserted_window_dims = [0],
       scatter_dims_to_operand_dims = [0],
       index_vector_dim = 1
     >
  } : (tensor<?xbf16>, tensor<?xi32>, tensor<?xbf16>) -> tensor<?xbf16>
  return %0 : tensor<?xbf16>
}
