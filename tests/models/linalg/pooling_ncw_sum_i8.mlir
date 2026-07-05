func.func @main(%arg0: tensor<?x?x?xi8>, %arg1: tensor<?xi8>) -> tensor<?x?x?xi8> {
  %c0 = arith.constant 0 : i8
  %c0_idx = arith.constant 0 : index
  %c1_idx = arith.constant 1 : index
  %c2_idx = arith.constant 2 : index

  %n = tensor.dim %arg0, %c0_idx : tensor<?x?x?xi8>
  %c = tensor.dim %arg0, %c1_idx : tensor<?x?x?xi8>
  %w = tensor.dim %arg0, %c2_idx : tensor<?x?x?xi8>

  %kw = tensor.dim %arg1, %c0_idx : tensor<?xi8>

  %subw = arith.subi %w, %kw : index
  %c1 = arith.constant 1 : index
  %ow = arith.addi %subw, %c1 : index

  %empty = tensor.empty(%n, %c, %ow) : tensor<?x?x?xi8>
  %fill = linalg.fill ins(%c0 : i8) outs(%empty : tensor<?x?x?xi8>) -> tensor<?x?x?xi8>
  %0 = linalg.pooling_ncw_sum
       {strides = dense<1> : vector<1xi64>, dilations = dense<1> : vector<1xi64>}
       ins(%arg0, %arg1 : tensor<?x?x?xi8>, tensor<?xi8>)
       outs(%fill : tensor<?x?x?xi8>) -> tensor<?x?x?xi8>
  return %0 : tensor<?x?x?xi8>
}
