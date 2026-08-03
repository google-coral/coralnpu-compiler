func.func @main(%arg0: tensor<?x?x?xi16>, %arg1: tensor<?x?xi16>) -> tensor<?x?x?xi16> {
  %c0 = arith.constant 0 : i16
  %c0_idx = arith.constant 0 : index
  %c1_idx = arith.constant 1 : index
  %c2_idx = arith.constant 2 : index

  %n = tensor.dim %arg0, %c0_idx : tensor<?x?x?xi16>
  %input_w = tensor.dim %arg0, %c1_idx : tensor<?x?x?xi16>
  %c = tensor.dim %arg0, %c2_idx : tensor<?x?x?xi16>

  %kernel_w = tensor.dim %arg1, %c0_idx : tensor<?x?xi16>

  %output_w_sub = arith.subi %input_w, %kernel_w : index

  %c1_output_w = arith.constant 1 : index

  %output_w = arith.addi %output_w_sub, %c1_output_w : index

  %empty = tensor.empty(%n, %output_w, %c) : tensor<?x?x?xi16>
  %fill = linalg.fill ins(%c0 : i16) outs(%empty : tensor<?x?x?xi16>) -> tensor<?x?x?xi16>
  %0 = linalg.depthwise_conv_1d_nwc_wc
       {strides = dense<1> : vector<1xi64>, dilations = dense<1> : vector<1xi64>}
       ins(%arg0, %arg1 : tensor<?x?x?xi16>, tensor<?x?xi16>)
       outs(%fill : tensor<?x?x?xi16>) -> tensor<?x?x?xi16>
  return %0 : tensor<?x?x?xi16>
}
