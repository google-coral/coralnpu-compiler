module {
  func.func @"compare_eq_rank1_i8_8-8"() {
    %cst = arith.constant dense<true> : tensor<8xi1>
    %cst_0 = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi8>
    %0 = util.optimization_barrier %cst_0 : tensor<8xi8>
    %1 = util.optimization_barrier %cst_0 : tensor<8xi8>
    %2 = stablehlo.compare EQ, %0, %1, SIGNED : (tensor<8xi8>, tensor<8xi8>) -> tensor<8xi1>
    %3 = util.optimization_barrier %cst : tensor<8xi1>
    "check.expect_eq"(%2, %3) : (tensor<8xi1>, tensor<8xi1>) -> ()
    return
  }
}
