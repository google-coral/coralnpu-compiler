module {
  func.func @fill_rank1_i8_8() {
    %cst = arith.constant dense<1> : tensor<8xi8>
    %c1_i8 = arith.constant 1 : i8
    %cst_0 = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi8>
    %0 = util.optimization_barrier %cst_0 : tensor<8xi8>
    %1 = linalg.fill ins(%c1_i8 : i8) outs(%0 : tensor<8xi8>) -> tensor<8xi8>
    %2 = util.optimization_barrier %cst : tensor<8xi8>
    "check.expect_eq"(%1, %2) : (tensor<8xi8>, tensor<8xi8>) -> ()
    return
  }
}
