module {
  func.func @"div_i8_4_8-4_8"() {
    %cst = arith.constant dense<1> : tensor<4x8xi8>
    %cst_0 = arith.constant dense<[[1, 2, 3, 4, 5, 1, 2, 3], [4, 5, 1, 2, 3, 4, 5, 1], [2, 3, 4, 5, 1, 2, 3, 4], [5, 1, 2, 3, 4, 5, 1, 2]]> : tensor<4x8xi8>
    %0 = util.optimization_barrier %cst_0 : tensor<4x8xi8>
    %1 = util.optimization_barrier %cst_0 : tensor<4x8xi8>
    %2 = tensor.empty() : tensor<4x8xi8>
    %3 = linalg.div ins(%0, %1 : tensor<4x8xi8>, tensor<4x8xi8>) outs(%2 : tensor<4x8xi8>) -> tensor<4x8xi8>
    %4 = util.optimization_barrier %cst : tensor<4x8xi8>
    "check.expect_eq"(%3, %4) : (tensor<4x8xi8>, tensor<4x8xi8>) -> ()
    return
  }
}
