module {
  func.func @fill_rank1_i32_8() {
    %cst = arith.constant dense<1> : tensor<8xi32>
    %c1_i32 = arith.constant 1 : i32
    %cst_0 = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi32>
    %0 = util.optimization_barrier %cst_0 : tensor<8xi32>
    %1 = linalg.fill ins(%c1_i32 : i32) outs(%0 : tensor<8xi32>) -> tensor<8xi32>
    %2 = util.optimization_barrier %cst : tensor<8xi32>
    "check.expect_eq"(%1, %2) : (tensor<8xi32>, tensor<8xi32>) -> ()
    return
  }
}
