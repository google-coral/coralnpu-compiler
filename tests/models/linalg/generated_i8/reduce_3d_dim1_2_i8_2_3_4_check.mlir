module {
  func.func @reduce_3d_dim1_2_i8_2_3_4() {
    %cst = arith.constant dense<[66, -46]> : tensor<2xi8>
    %c0_i8 = arith.constant 0 : i8
    %cst_0 = arith.constant dense<[[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]> : tensor<2x3x4xi8>
    %0 = util.optimization_barrier %cst_0 : tensor<2x3x4xi8>
    %1 = tensor.empty() : tensor<2xi8>
    %2 = linalg.fill ins(%c0_i8 : i8) outs(%1 : tensor<2xi8>) -> tensor<2xi8>
    %reduced = linalg.reduce ins(%0 : tensor<2x3x4xi8>) outs(%2 : tensor<2xi8>) dimensions = [1, 2] 
      (%in: i8, %init: i8) {
        %4 = arith.addi %in, %init : i8
        linalg.yield %4 : i8
      }
    %3 = util.optimization_barrier %cst : tensor<2xi8>
    "check.expect_eq"(%reduced, %3) : (tensor<2xi8>, tensor<2xi8>) -> ()
    return
  }
}
