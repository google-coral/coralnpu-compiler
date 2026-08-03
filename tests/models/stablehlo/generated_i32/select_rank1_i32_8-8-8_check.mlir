module {
  func.func @"select_rank1_i32_8-8-8"() {
    %cst = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi32>
    %cst_0 = arith.constant dense<[false, true, false, true, false, true, false, true]> : tensor<8xi1>
    %0 = util.optimization_barrier %cst_0 : tensor<8xi1>
    %1 = util.optimization_barrier %cst : tensor<8xi32>
    %2 = util.optimization_barrier %cst : tensor<8xi32>
    %3 = stablehlo.select %0, %1, %2 : tensor<8xi1>, tensor<8xi32>
    %4 = util.optimization_barrier %cst : tensor<8xi32>
    "check.expect_eq"(%3, %4) : (tensor<8xi32>, tensor<8xi32>) -> ()
    return
  }
}
