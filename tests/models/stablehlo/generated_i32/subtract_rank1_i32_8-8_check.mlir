module {
  func.func @"subtract_rank1_i32_8-8"() {
    %cst = arith.constant dense<0> : tensor<8xi32>
    %cst_0 = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi32>
    %0 = util.optimization_barrier %cst_0 : tensor<8xi32>
    %1 = util.optimization_barrier %cst_0 : tensor<8xi32>
    %2 = stablehlo.subtract %0, %1 : tensor<8xi32>
    %3 = util.optimization_barrier %cst : tensor<8xi32>
    "check.expect_eq"(%2, %3) : (tensor<8xi32>, tensor<8xi32>) -> ()
    return
  }
}
