module {
  func.func @"dot_i8_8-8"() {
    %cst = arith.constant dense<-116> : tensor<i8>
    %c0_i8 = arith.constant 0 : i8
    %cst_0 = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi8>
    %0 = util.optimization_barrier %cst_0 : tensor<8xi8>
    %1 = util.optimization_barrier %cst_0 : tensor<8xi8>
    %2 = tensor.empty() : tensor<i8>
    %3 = linalg.fill ins(%c0_i8 : i8) outs(%2 : tensor<i8>) -> tensor<i8>
    %4 = linalg.dot ins(%0, %1 : tensor<8xi8>, tensor<8xi8>) outs(%3 : tensor<i8>) -> tensor<i8>
    %5 = util.optimization_barrier %cst : tensor<i8>
    "check.expect_eq"(%4, %5) : (tensor<i8>, tensor<i8>) -> ()
    return
  }
}
