func.func @main(%arg0: tensor<?x?x?x?xi16>, %arg1: tensor<?x?x?xi16>) -> tensor<?x?x?x?xi16> {
  %c0 = arith.constant 0 : i16
  %c0_idx = arith.constant 0 : index
  %c1_idx = arith.constant 1 : index
  %c2_idx = arith.constant 2 : index
  %c3_idx = arith.constant 3 : index
  %n = tensor.dim %arg0, %c0_idx : tensor<?x?x?x?xi16>
  %h = tensor.dim %arg0, %c1_idx : tensor<?x?x?x?xi16>
  %w = tensor.dim %arg0, %c2_idx : tensor<?x?x?x?xi16>
  %c = tensor.dim %arg0, %c3_idx : tensor<?x?x?x?xi16>
  %kh = tensor.dim %arg1, %c0_idx : tensor<?x?x?xi16>
  %kw = tensor.dim %arg1, %c1_idx : tensor<?x?x?xi16>

  %subh = arith.subi %h, %kh : index
  %c1 = arith.constant 1 : index
  %oh = arith.addi %subh, %c1 : index

  %subw = arith.subi %w, %kw : index
  %ow = arith.addi %subw, %c1 : index

  %empty = tensor.empty(%n, %oh, %ow, %c) : tensor<?x?x?x?xi16>
  %fill = linalg.fill ins(%c0 : i16) outs(%empty : tensor<?x?x?x?xi16>) -> tensor<?x?x?x?xi16>
  %0 = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %arg1 : tensor<?x?x?x?xi16>, tensor<?x?x?xi16>) outs(%fill : tensor<?x?x?x?xi16>) -> tensor<?x?x?x?xi16>
  return %0 : tensor<?x?x?x?xi16>
}
