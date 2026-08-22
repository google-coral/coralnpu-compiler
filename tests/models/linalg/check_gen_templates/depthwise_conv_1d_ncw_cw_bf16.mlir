func.func @main(%arg0: tensor<?x?x?xbf16>, %arg1: tensor<?x?xbf16>) -> tensor<?x?x?xbf16> {
  %c0 = arith.constant 0.0 : bf16
  %c0_idx = arith.constant 0 : index
  %c1_idx = arith.constant 1 : index
  %c2_idx = arith.constant 2 : index

  %n = tensor.dim %arg0, %c0_idx : tensor<?x?x?xbf16>
  %c = tensor.dim %arg0, %c1_idx : tensor<?x?x?xbf16>
  %w = tensor.dim %arg0, %c2_idx : tensor<?x?x?xbf16>

  %kw = tensor.dim %arg1, %c1_idx : tensor<?x?xbf16>

  %subw = arith.subi %w, %kw : index
  %c1 = arith.constant 1 : index
  %ow = arith.addi %subw, %c1 : index

  %empty = tensor.empty(%n, %c, %ow) : tensor<?x?x?xbf16>
  %fill = linalg.fill ins(%c0 : bf16) outs(%empty : tensor<?x?x?xbf16>) -> tensor<?x?x?xbf16>
  %0 = linalg.depthwise_conv_1d_ncw_cw
       {strides = dense<1> : vector<1xi64>, dilations = dense<1> : vector<1xi64>}
       ins(%arg0, %arg1 : tensor<?x?x?xbf16>, tensor<?x?xbf16>)
       outs(%fill : tensor<?x?x?xbf16>) -> tensor<?x?x?xbf16>
  return %0 : tensor<?x?x?xbf16>
}
