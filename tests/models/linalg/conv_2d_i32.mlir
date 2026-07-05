func.func @main(%arg0: tensor<?x?xi32>, %arg1: tensor<?x?xi32>) -> tensor<?x?xi32> {
  %c0 = arith.constant 0 : i32
  %c0_idx = arith.constant 0 : index
  %c1_idx = arith.constant 1 : index

  %input_h = tensor.dim %arg0, %c0_idx : tensor<?x?xi32>
  %input_w = tensor.dim %arg0, %c1_idx : tensor<?x?xi32>

  %kernel_h = tensor.dim %arg1, %c0_idx : tensor<?x?xi32>
  %kernel_w = tensor.dim %arg1, %c1_idx : tensor<?x?xi32>

  %output_h_sub = arith.subi %input_h, %kernel_h : index

  %c1_output_h = arith.constant 1 : index

  %output_h = arith.addi %output_h_sub, %c1_output_h : index
  %output_w_sub = arith.subi %input_w, %kernel_w : index
  %c1_output_w = arith.constant 1 : index
  %output_w = arith.addi %output_w_sub, %c1_output_w : index

  %empty = tensor.empty(%output_h, %output_w) : tensor<?x?xi32>
  %fill = linalg.fill ins(%c0 : i32) outs(%empty : tensor<?x?xi32>) -> tensor<?x?xi32>
  %0 = linalg.conv_2d
       ins(%arg0, %arg1 : tensor<?x?xi32>, tensor<?x?xi32>)
       outs(%fill : tensor<?x?xi32>) -> tensor<?x?xi32>
  return %0 : tensor<?x?xi32>
}
