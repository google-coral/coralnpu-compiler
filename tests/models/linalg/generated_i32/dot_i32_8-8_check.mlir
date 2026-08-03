module {
  func.func @"dot_i32_8-8"() {
    %cst = arith.constant dense<140> : tensor<i32>
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi32>
    %0 = util.optimization_barrier %cst_0 : tensor<8xi32>
    %1 = util.optimization_barrier %cst_0 : tensor<8xi32>
    %2 = tensor.empty() : tensor<i32>
    %3 = linalg.fill ins(%c0_i32 : i32) outs(%2 : tensor<i32>) -> tensor<i32>
    %4 = linalg.dot ins(%0, %1 : tensor<8xi32>, tensor<8xi32>) outs(%3 : tensor<i32>) -> tensor<i32>
    %5 = util.optimization_barrier %cst : tensor<i32>
    "check.expect_eq"(%4, %5) : (tensor<i32>, tensor<i32>) -> ()
    return
  }
}
