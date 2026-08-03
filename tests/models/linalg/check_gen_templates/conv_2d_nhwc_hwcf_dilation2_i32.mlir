func.func @main(%arg0: tensor<?x?x?x?xi32>, %arg1: tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32> {
  %c0 = arith.constant 0 : i32
  %c0_idx = arith.constant 0 : index
  %c1_idx = arith.constant 1 : index
  %c2_idx = arith.constant 2 : index
  %c3_idx = arith.constant 3 : index
  %n = tensor.dim %arg0, %c0_idx : tensor<?x?x?x?xi32>
  %h = tensor.dim %arg0, %c1_idx : tensor<?x?x?x?xi32>
  %w = tensor.dim %arg0, %c2_idx : tensor<?x?x?x?xi32>
  %kh = tensor.dim %arg1, %c0_idx : tensor<?x?x?x?xi32>
  %kw = tensor.dim %arg1, %c1_idx : tensor<?x?x?x?xi32>
  %f = tensor.dim %arg1, %c3_idx : tensor<?x?x?x?xi32>

  %c2_oh = arith.constant 2 : index
  %mul_oh = arith.muli %kh, %c2_oh : index
  %subh = arith.subi %h, %mul_oh : index
  %c1 = arith.constant 1 : index
  %oh = arith.addi %subh, %c2_oh : index

  %c2_ow = arith.constant 2 : index
  %mul_ow = arith.muli %kw, %c2_ow : index
  %subw = arith.subi %w, %mul_ow : index
  %ow = arith.addi %subw, %c2_ow : index

  %empty = tensor.empty(%n, %oh, %ow, %f) : tensor<?x?x?x?xi32>
  %fill = linalg.fill ins(%c0 : i32) outs(%empty : tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %0 = linalg.conv_2d_nhwc_hwcf {dilations = dense<2> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %arg1 : tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) outs(%fill : tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  return %0 : tensor<?x?x?x?xi32>
}
