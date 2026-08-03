module {
  func.func @reduce_3d_dim1_2_i16_2_3_4() {
    %cst = arith.constant dense<[66, 210]> : tensor<2xi16>
    %c0_i16 = arith.constant 0 : i16
    %cst_0 = arith.constant dense<[[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]> : tensor<2x3x4xi16>
    %0 = util.optimization_barrier %cst_0 : tensor<2x3x4xi16>
    %1 = tensor.empty() : tensor<2xi16>
    %2 = linalg.fill ins(%c0_i16 : i16) outs(%1 : tensor<2xi16>) -> tensor<2xi16>
    %reduced = linalg.reduce ins(%0 : tensor<2x3x4xi16>) outs(%2 : tensor<2xi16>) dimensions = [1, 2] 
      (%in: i16, %init: i16) {
        %4 = arith.addi %in, %init : i16
        linalg.yield %4 : i16
      }
    %3 = util.optimization_barrier %cst : tensor<2xi16>
    "check.expect_eq"(%reduced, %3) : (tensor<2xi16>, tensor<2xi16>) -> ()
    return
  }
}
