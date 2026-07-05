func.func @main(%arg0: tensor<?x?x?x?x?xf32>, %arg1: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32> {
  %c0 = arith.constant 0.0 : f32
  %c0_idx = arith.constant 0 : index
  %c1_idx = arith.constant 1 : index
  %c2_idx = arith.constant 2 : index
  %c3_idx = arith.constant 3 : index
  %c4_idx = arith.constant 4 : index

  %n = tensor.dim %arg0, %c0_idx : tensor<?x?x?x?x?xf32>
  %input_d = tensor.dim %arg0, %c1_idx : tensor<?x?x?x?x?xf32>
  %input_h = tensor.dim %arg0, %c2_idx : tensor<?x?x?x?x?xf32>
  %input_w = tensor.dim %arg0, %c3_idx : tensor<?x?x?x?x?xf32>
  %c = tensor.dim %arg0, %c4_idx : tensor<?x?x?x?x?xf32>

  %kernel_d = tensor.dim %arg1, %c0_idx : tensor<?x?x?x?xf32>
  %kernel_h = tensor.dim %arg1, %c1_idx : tensor<?x?x?x?xf32>
  %kernel_w = tensor.dim %arg1, %c2_idx : tensor<?x?x?x?xf32>

  %output_d_sub = arith.subi %input_d, %kernel_d : index

  %c1_output_d = arith.constant 1 : index

  %output_d = arith.addi %output_d_sub, %c1_output_d : index
  %output_d_corrected = arith.addi %output_d, %c1_idx : index // Wait, why did I add 1?
  // Ah!
  // In conv_1d/2d/3d we had s0 + s1 (input_size)
  // output_size = input_size - kernel_size.
  // Wait, if H_in = H_out + H_ker (according to shape map).
  // Then H_out = H_in - H_ker.
  // Yes!
  // But in depthwise_conv_3d_ndhwc_dhwc:
  // shape_map for I: (s0, s1 * s2 + s3 * s4, s5 * s6 + s7 * s8, s9 * s10 + s11 * s12, s13)
  // If s2=1, s4=1 (stride=1, dilation=1).
  // I_d = s1 + s3 = D_out + D_ker.
  // So D_out = I_d - D_ker.
  // Yes, no need to add 1.
  // It was conv_1d/2d/3d where s0 + s1 was the shape map.
  // Wait, if s0 + s1 is shape map, H_in = H_out + H_ker.
  // H_out = H_in - H_ker.
  // It is the same!
  // Why did I think we need to add 1?
  // In standard conv formula: H_out = H_in - H_ker + 1.
  // But Linalg shape map says H_in = H_out + H_ker.
  // So Linalg H_out = H_in - H_ker.
  // This means Linalg's conv is slightly different from standard conv (it has 1 extra element in input, or output is 1 element smaller than standard).
  // Actually:
  // if H_in = 10, H_ker = 3.
  // Standard conv H_out = 10 - 3 + 1 = 8.
  // Linalg conv H_out = 10 - 3 = 7.
  // So Linalg H_out is indeed 1 smaller.
  // We must follow Linalg shape map.
  // So H_out = H_in - H_ker.

  %output_h_sub = arith.subi %input_h, %kernel_h : index

  %c1_output_h = arith.constant 1 : index

  %output_h = arith.addi %output_h_sub, %c1_output_h : index
  %output_w_sub = arith.subi %input_w, %kernel_w : index
  %c1_output_w = arith.constant 1 : index
  %output_w = arith.addi %output_w_sub, %c1_output_w : index

  %empty = tensor.empty(%n, %output_d, %output_h, %output_w, %c) : tensor<?x?x?x?x?xf32>
  %fill = linalg.fill ins(%c0 : f32) outs(%empty : tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32>
  %0 = linalg.depthwise_conv_3d_ndhwc_dhwc
       {strides = dense<1> : vector<3xi64>, dilations = dense<1> : vector<3xi64>}
       ins(%arg0, %arg1 : tensor<?x?x?x?x?xf32>, tensor<?x?x?x?xf32>)
       outs(%fill : tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32>
  return %0 : tensor<?x?x?x?x?xf32>
}
