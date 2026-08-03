module {
  func.func @transpose_3d_i8_2_3_4() {
    %cst = arith.constant dense<[[[0, 4, 8], [1, 5, 9], [2, 6, 10], [3, 7, 11]], [[12, 16, 20], [13, 17, 21], [14, 18, 22], [15, 19, 23]]]> : tensor<2x4x3xi8>
    %cst_0 = arith.constant dense<[[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]> : tensor<2x3x4xi8>
    %0 = util.optimization_barrier %cst_0 : tensor<2x3x4xi8>
    %1 = tensor.empty() : tensor<2x4x3xi8>
    %transposed = linalg.transpose ins(%0 : tensor<2x3x4xi8>) outs(%1 : tensor<2x4x3xi8>) permutation = [0, 2, 1] 
    %2 = util.optimization_barrier %cst : tensor<2x4x3xi8>
    "check.expect_eq"(%transposed, %2) : (tensor<2x4x3xi8>, tensor<2x4x3xi8>) -> ()
    return
  }
}
