module {
  func.func @"div_i32_4_8-4_8"() {
    %cst = arith.constant dense<1> : tensor<4x8xi32>
    %cst_0 = arith.constant dense<[[1, 2, 3, 4, 5, 1, 2, 3], [4, 5, 1, 2, 3, 4, 5, 1], [2, 3, 4, 5, 1, 2, 3, 4], [5, 1, 2, 3, 4, 5, 1, 2]]> : tensor<4x8xi32>
    %0 = util.optimization_barrier %cst_0 : tensor<4x8xi32>
    %1 = util.optimization_barrier %cst_0 : tensor<4x8xi32>
    %2 = tensor.empty() : tensor<4x8xi32>
    %3 = linalg.div ins(%0, %1 : tensor<4x8xi32>, tensor<4x8xi32>) outs(%2 : tensor<4x8xi32>) -> tensor<4x8xi32>
    %4 = util.optimization_barrier %cst : tensor<4x8xi32>
    "check.expect_eq"(%3, %4) : (tensor<4x8xi32>, tensor<4x8xi32>) -> ()
    return
  }
}
