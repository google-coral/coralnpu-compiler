module {
  func.func @"dot_i16_8-8"() {
    %cst = arith.constant dense<140> : tensor<i16>
    %c0_i16 = arith.constant 0 : i16
    %cst_0 = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi16>
    %0 = util.optimization_barrier %cst_0 : tensor<8xi16>
    %1 = util.optimization_barrier %cst_0 : tensor<8xi16>
    %2 = tensor.empty() : tensor<i16>
    %3 = linalg.fill ins(%c0_i16 : i16) outs(%2 : tensor<i16>) -> tensor<i16>
    %4 = linalg.dot ins(%0, %1 : tensor<8xi16>, tensor<8xi16>) outs(%3 : tensor<i16>) -> tensor<i16>
    %5 = util.optimization_barrier %cst : tensor<i16>
    "check.expect_eq"(%4, %5) : (tensor<i16>, tensor<i16>) -> ()
    return
  }
}
