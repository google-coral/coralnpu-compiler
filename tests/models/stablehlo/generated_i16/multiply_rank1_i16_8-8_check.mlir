module {
  func.func @"multiply_rank1_i16_8-8"() {
    %cst = arith.constant dense<[0, 1, 4, 9, 16, 25, 36, 49]> : tensor<8xi16>
    %cst_0 = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi16>
    %0 = util.optimization_barrier %cst_0 : tensor<8xi16>
    %1 = util.optimization_barrier %cst_0 : tensor<8xi16>
    %2 = stablehlo.multiply %0, %1 : tensor<8xi16>
    %3 = util.optimization_barrier %cst : tensor<8xi16>
    "check.expect_eq"(%2, %3) : (tensor<8xi16>, tensor<8xi16>) -> ()
    return
  }
}
