module {
  func.func @reduce_3d_dim1_2_i32_2_3_4() {
    %cst = arith.constant dense<[66, 210]> : tensor<2xi32>
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant dense<[[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]> : tensor<2x3x4xi32>
    %0 = util.optimization_barrier %cst_0 : tensor<2x3x4xi32>
    %1 = tensor.empty() : tensor<2xi32>
    %2 = linalg.fill ins(%c0_i32 : i32) outs(%1 : tensor<2xi32>) -> tensor<2xi32>
    %reduced = linalg.reduce ins(%0 : tensor<2x3x4xi32>) outs(%2 : tensor<2xi32>) dimensions = [1, 2] 
      (%in: i32, %init: i32) {
        %4 = arith.addi %in, %init : i32
        linalg.yield %4 : i32
      }
    %3 = util.optimization_barrier %cst : tensor<2xi32>
    "check.expect_eq"(%reduced, %3) : (tensor<2xi32>, tensor<2xi32>) -> ()
    return
  }
}
