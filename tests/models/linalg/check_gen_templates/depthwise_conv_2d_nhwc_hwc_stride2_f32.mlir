func.func @main(%arg0: tensor<?x?x?x?xf32>, %arg1: tensor<?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %c0 = arith.constant 0.0 : f32
  %c0_idx = arith.constant 0 : index
  %c1_idx = arith.constant 1 : index
  %c2_idx = arith.constant 2 : index
  %c3_idx = arith.constant 3 : index
  %n = tensor.dim %arg0, %c0_idx : tensor<?x?x?x?xf32>
  %h = tensor.dim %arg0, %c1_idx : tensor<?x?x?x?xf32>
  %w = tensor.dim %arg0, %c2_idx : tensor<?x?x?x?xf32>
  %c = tensor.dim %arg0, %c3_idx : tensor<?x?x?x?xf32>
  %kh = tensor.dim %arg1, %c0_idx : tensor<?x?x?xf32>
  %kw = tensor.dim %arg1, %c1_idx : tensor<?x?x?xf32>

  %subh = arith.subi %h, %kh : index
  %c1 = arith.constant 1 : index
  %c2_oh = arith.constant 2 : index
  %oh_div = arith.divsi %subh, %c2_oh : index
  %oh = arith.addi %oh_div, %c1 : index

  %subw = arith.subi %w, %kw : index
  %c2_ow = arith.constant 2 : index
  %ow_div = arith.divsi %subw, %c2_ow : index
  %ow = arith.addi %ow_div, %c1 : index

  %empty = tensor.empty(%n, %oh, %ow, %c) : tensor<?x?x?x?xf32>
  %fill = linalg.fill ins(%c0 : f32) outs(%empty : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %0 = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%arg0, %arg1 : tensor<?x?x?x?xf32>, tensor<?x?x?xf32>) outs(%fill : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}
