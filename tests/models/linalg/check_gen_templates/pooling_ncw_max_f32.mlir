func.func @main(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?xf32>) -> tensor<?x?x?xf32> {
  %min_f32 = arith.constant -3.40282347E+38 : f32
  %c0_idx = arith.constant 0 : index
  %c1_idx = arith.constant 1 : index
  %c2_idx = arith.constant 2 : index

  %n = tensor.dim %arg0, %c0_idx : tensor<?x?x?xf32>
  %c = tensor.dim %arg0, %c1_idx : tensor<?x?x?xf32>
  %w = tensor.dim %arg0, %c2_idx : tensor<?x?x?xf32>

  %kw = tensor.dim %arg1, %c0_idx : tensor<?xf32>

  %subw = arith.subi %w, %kw : index
  %c1 = arith.constant 1 : index
  %ow = arith.addi %subw, %c1 : index

  %empty = tensor.empty(%n, %c, %ow) : tensor<?x?x?xf32>
  %fill = linalg.fill ins(%min_f32 : f32) outs(%empty : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %0 = linalg.pooling_ncw_max
       {strides = dense<1> : vector<1xi64>, dilations = dense<1> : vector<1xi64>}
       ins(%arg0, %arg1 : tensor<?x?x?xf32>, tensor<?xf32>)
       outs(%fill : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
}
