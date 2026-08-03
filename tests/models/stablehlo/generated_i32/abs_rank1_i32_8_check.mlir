module {
  func.func @abs_rank1_i32_8() {
    %cst = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi32>
    %0 = util.optimization_barrier %cst : tensor<8xi32>
    %1 = stablehlo.abs %0 : tensor<8xi32>
    %2 = util.optimization_barrier %cst : tensor<8xi32>
    "check.expect_eq"(%1, %2) : (tensor<8xi32>, tensor<8xi32>) -> ()
    return
  }
}
