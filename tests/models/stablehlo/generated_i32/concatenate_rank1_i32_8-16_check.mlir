module {
  func.func @"concatenate_rank1_i32_8-16"() {
    %cst = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : tensor<24xi32>
    %cst_0 = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : tensor<16xi32>
    %cst_1 = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi32>
    %0 = util.optimization_barrier %cst_1 : tensor<8xi32>
    %1 = util.optimization_barrier %cst_0 : tensor<16xi32>
    %2 = stablehlo.concatenate %0, %1, dim = 0 : (tensor<8xi32>, tensor<16xi32>) -> tensor<24xi32>
    %3 = util.optimization_barrier %cst : tensor<24xi32>
    "check.expect_eq"(%2, %3) : (tensor<24xi32>, tensor<24xi32>) -> ()
    return
  }
}
