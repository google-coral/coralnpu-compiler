module {
  func.func @"shift_right_logical_rank1_i8_8-8"() {
    %cst = arith.constant dense<0> : tensor<8xi8>
    %cst_0 = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi8>
    %0 = util.optimization_barrier %cst_0 : tensor<8xi8>
    %1 = util.optimization_barrier %cst_0 : tensor<8xi8>
    %2 = stablehlo.shift_right_logical %0, %1 : tensor<8xi8>
    %3 = util.optimization_barrier %cst : tensor<8xi8>
    "check.expect_eq"(%2, %3) : (tensor<8xi8>, tensor<8xi8>) -> ()
    return
  }
}
