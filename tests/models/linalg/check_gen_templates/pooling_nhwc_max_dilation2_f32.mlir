func.func @main(%arg0: tensor<?x?x?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?x?x?xf32> {
  %min_f32 = arith.constant -3.40282347E+38 : f32
  %c0_idx = arith.constant 0 : index
  %c1_idx = arith.constant 1 : index
  %c2_idx = arith.constant 2 : index
  %c3_idx = arith.constant 3 : index

  %n = tensor.dim %arg0, %c0_idx : tensor<?x?x?x?xf32>
  %input_h = tensor.dim %arg0, %c1_idx : tensor<?x?x?x?xf32>
  %input_w = tensor.dim %arg0, %c2_idx : tensor<?x?x?x?xf32>
  %c = tensor.dim %arg0, %c3_idx : tensor<?x?x?x?xf32>

  %kernel_h = tensor.dim %arg1, %c0_idx : tensor<?x?xf32>
  %kernel_w = tensor.dim %arg1, %c1_idx : tensor<?x?xf32>

  %c2_output_h = arith.constant 2 : index
  %mul_output_h = arith.muli %kernel_h, %c2_output_h : index
  %output_h_sub = arith.subi %input_h, %mul_output_h : index

  %c1_output_h = arith.constant 1 : index

  %output_h = arith.addi %output_h_sub, %c2_output_h : index
  %c2_output_w = arith.constant 2 : index
  %mul_output_w = arith.muli %kernel_w, %c2_output_w : index
  %output_w_sub = arith.subi %input_w, %mul_output_w : index
  %c1_output_w = arith.constant 1 : index
  %output_w = arith.addi %output_w_sub, %c2_output_w : index

  %empty = tensor.empty(%n, %output_h, %output_w, %c) : tensor<?x?x?x?xf32>
  %fill = linalg.fill ins(%min_f32 : f32) outs(%empty : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %0 = linalg.pooling_nhwc_max
       {strides = dense<1> : vector<2xi64>, dilations = dense<2> : vector<2xi64>}
       ins(%arg0, %arg1 : tensor<?x?x?x?xf32>, tensor<?x?xf32>)
       outs(%fill : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}
