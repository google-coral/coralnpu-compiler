func.func @main(%arg0: tensor<?x?x?x?x?xi8>, %arg1: tensor<?x?x?xi8>) -> tensor<?x?x?x?x?xi8> {
  %c0 = arith.constant 127 : i8
  %c0_idx = arith.constant 0 : index
  %c1_idx = arith.constant 1 : index
  %c2_idx = arith.constant 2 : index
  %c3_idx = arith.constant 3 : index
  %c4_idx = arith.constant 4 : index

  %n = tensor.dim %arg0, %c0_idx : tensor<?x?x?x?x?xi8>
  %input_d = tensor.dim %arg0, %c1_idx : tensor<?x?x?x?x?xi8>
  %input_h = tensor.dim %arg0, %c2_idx : tensor<?x?x?x?x?xi8>
  %input_w = tensor.dim %arg0, %c3_idx : tensor<?x?x?x?x?xi8>
  %c = tensor.dim %arg0, %c4_idx : tensor<?x?x?x?x?xi8>

  %kernel_d = tensor.dim %arg1, %c0_idx : tensor<?x?x?xi8>
  %kernel_h = tensor.dim %arg1, %c1_idx : tensor<?x?x?xi8>
  %kernel_w = tensor.dim %arg1, %c2_idx : tensor<?x?x?xi8>

  %output_d_sub = arith.subi %input_d, %kernel_d : index

  %c1_output_d = arith.constant 1 : index

  %output_d = arith.addi %output_d_sub, %c1_output_d : index
  %output_h_sub = arith.subi %input_h, %kernel_h : index
  %c1_output_h = arith.constant 1 : index
  %output_h = arith.addi %output_h_sub, %c1_output_h : index
  %output_w_sub = arith.subi %input_w, %kernel_w : index
  %c1_output_w = arith.constant 1 : index
  %output_w = arith.addi %output_w_sub, %c1_output_w : index

  %empty = tensor.empty(%n, %output_d, %output_h, %output_w, %c) : tensor<?x?x?x?x?xi8>
  %fill = linalg.fill ins(%c0 : i8) outs(%empty : tensor<?x?x?x?x?xi8>) -> tensor<?x?x?x?x?xi8>
  %0 = linalg.pooling_ndhwc_min
       {strides = dense<1> : vector<3xi64>, dilations = dense<1> : vector<3xi64>}
       ins(%arg0, %arg1 : tensor<?x?x?x?x?xi8>, tensor<?x?x?xi8>)
       outs(%fill : tensor<?x?x?x?x?xi8>) -> tensor<?x?x?x?x?xi8>
  return %0 : tensor<?x?x?x?x?xi8>
}
