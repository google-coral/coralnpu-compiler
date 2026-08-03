func.func @main(%arg0: tensor<?xi32>, %arg1: tensor<?xi32>) -> tensor<?xi32> {
  %c0 = arith.constant 0 : i32
  %c0_idx = arith.constant 0 : index
  %input_size = tensor.dim %arg0, %c0_idx : tensor<?xi32>
  %kernel_size = tensor.dim %arg1, %c0_idx : tensor<?xi32>
  %output_size_sub = arith.subi %input_size, %kernel_size : index
  %c1_output_size = arith.constant 1 : index
  %output_size = arith.addi %output_size_sub, %c1_output_size : index

  %empty = tensor.empty(%output_size) : tensor<?xi32>
  %fill = linalg.fill ins(%c0 : i32) outs(%empty : tensor<?xi32>) -> tensor<?xi32>
  %0 = linalg.conv_1d
       ins(%arg0, %arg1 : tensor<?xi32>, tensor<?xi32>)
       outs(%fill : tensor<?xi32>) -> tensor<?xi32>
  return %0 : tensor<?xi32>
}
