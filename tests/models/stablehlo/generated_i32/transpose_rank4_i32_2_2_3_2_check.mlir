module {
  func.func @transpose_rank4_i32_2_2_3_2() {
    %cst = arith.constant dense<[[[[0, 12], [1, 13]], [[2, 14], [3, 15]], [[4, 16], [5, 17]]], [[[6, 18], [7, 19]], [[8, 20], [9, 21]], [[10, 22], [11, 23]]]]> : tensor<2x3x2x2xi32>
    %cst_0 = arith.constant dense<[[[[0, 1], [2, 3], [4, 5]], [[6, 7], [8, 9], [10, 11]]], [[[12, 13], [14, 15], [16, 17]], [[18, 19], [20, 21], [22, 23]]]]> : tensor<2x2x3x2xi32>
    %0 = util.optimization_barrier %cst_0 : tensor<2x2x3x2xi32>
    %1 = stablehlo.transpose %0, dims = [1, 2, 3, 0] : (tensor<2x2x3x2xi32>) -> tensor<2x3x2x2xi32>
    %2 = util.optimization_barrier %cst : tensor<2x3x2x2xi32>
    "check.expect_eq"(%1, %2) : (tensor<2x3x2x2xi32>, tensor<2x3x2x2xi32>) -> ()
    return
  }
}
