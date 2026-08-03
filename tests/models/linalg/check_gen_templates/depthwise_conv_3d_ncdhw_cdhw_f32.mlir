// RUN: %template_path

func.func @main(%arg0: tensor<?x?x?x?x?xf32>, %arg1: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32> {
  %c0 = arith.constant 0.0 : f32
  %c0_idx = arith.constant 0 : index
  %c1_idx = arith.constant 1 : index
  %c2_idx = arith.constant 2 : index
  %c3_idx = arith.constant 3 : index
  %c4_idx = arith.constant 4 : index

  %n = tensor.dim %arg0, %c0_idx : tensor<?x?x?x?x?xf32>
  %c = tensor.dim %arg0, %c1_idx : tensor<?x?x?x?x?xf32>
  %input_d = tensor.dim %arg0, %c2_idx : tensor<?x?x?x?x?xf32>
  %input_h = tensor.dim %arg0, %c3_idx : tensor<?x?x?x?x?xf32>
  %input_w = tensor.dim %arg0, %c4_idx : tensor<?x?x?x?x?xf32>

  %kernel_d = tensor.dim %arg1, %c1_idx : tensor<?x?x?x?xf32>
  %kernel_h = tensor.dim %arg1, %c2_idx : tensor<?x?x?x?xf32>
  %kernel_w = tensor.dim %arg1, %c3_idx : tensor<?x?x?x?xf32>

  %output_d = arith.subi %input_d, %kernel_d : index
  %output_h = arith.subi %input_h, %kernel_h : index
  %output_w = arith.subi %input_w, %kernel_w : index

  %c1 = arith.constant 1 : index
  %od = arith.addi %output_d, %c1 : index
  %oh = arith.addi %output_h, %c1 : index
  %ow = arith.addi %output_w, %c1 : index

  %empty = tensor.empty(%n, %c, %od, %oh, %ow) : tensor<?x?x?x?x?xf32>
  %fill = linalg.fill ins(%c0 : f32) outs(%empty : tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32>

  %0 = linalg.depthwise_conv_3d_ncdhw_cdhw
       {strides = dense<1> : vector<3xi64>, dilations = dense<1> : vector<3xi64>}
       ins(%arg0, %arg1 : tensor<?x?x?x?x?xf32>, tensor<?x?x?x?xf32>)
       outs(%fill : tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32>
  return %0 : tensor<?x?x?x?x?xf32>
}
