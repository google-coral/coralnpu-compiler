module {
  func.func @"matmul_rank1_i16_8-8"() {
    %cst = arith.constant dense<140> : tensor<i16>
    %cst_0 = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi16>
    %0 = util.optimization_barrier %cst_0 : tensor<8xi16>
    %1 = util.optimization_barrier %cst_0 : tensor<8xi16>
    %2 = stablehlo.dot %0, %1 : (tensor<8xi16>, tensor<8xi16>) -> tensor<i16>
    %3 = util.optimization_barrier %cst : tensor<i16>
    "check.expect_eq"(%2, %3) : (tensor<i16>, tensor<i16>) -> ()
    return
  }
}
