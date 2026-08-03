func.func @main(%input: tensor<?xf32>, %scatter_indices: tensor<?xi32>, %updates: tensor<?xf32>) -> tensor<?xf32> {
  %0 = "stablehlo.scatter"(%input, %scatter_indices, %updates) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
     %res = stablehlo.add %lhs, %rhs : tensor<f32>
     "stablehlo.return"(%res) : (tensor<f32>) -> ()
  }) {
     indices_are_sorted = false,
     scatter_dimension_numbers = #stablehlo.scatter<
       update_window_dims = [],
       inserted_window_dims = [0],
       scatter_dims_to_operand_dims = [0],
       index_vector_dim = 1
     >
  } : (tensor<?xf32>, tensor<?xi32>, tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}
