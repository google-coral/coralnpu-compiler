func.func @main(%arg0: tensor<?xi16>, %arg1: tensor<?xi16>) -> tensor<?xi16> {
  %c0 = arith.constant 0 : i16
  %c0_idx = arith.constant 0 : index
  %input_size = tensor.dim %arg0, %c0_idx : tensor<?xi16>
  %kernel_size = tensor.dim %arg1, %c0_idx : tensor<?xi16>
  %output_size_sub = arith.subi %input_size, %kernel_size : index
  %c1_output_size = arith.constant 1 : index
  %output_size = arith.addi %output_size_sub, %c1_output_size : index

  %empty = tensor.empty(%output_size) : tensor<?xi16>
  %fill = linalg.fill ins(%c0 : i16) outs(%empty : tensor<?xi16>) -> tensor<?xi16>
  %0 = linalg.conv_1d
       ins(%arg0, %arg1 : tensor<?xi16>, tensor<?xi16>)
       outs(%fill : tensor<?xi16>) -> tensor<?xi16>
  return %0 : tensor<?xi16>
}
