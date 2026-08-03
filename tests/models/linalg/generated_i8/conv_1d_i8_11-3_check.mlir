module {
  func.func @"conv_1d_i8_11-3"() {
    %cst = arith.constant dense<[5, 8, 11, 14, 17, 20, 23, 26, 29]> : tensor<9xi8>
    %cst_0 = arith.constant dense<[0, 1, 2]> : tensor<3xi8>
    %c0_i8 = arith.constant 0 : i8
    %cst_1 = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]> : tensor<11xi8>
    %0 = util.optimization_barrier %cst_1 : tensor<11xi8>
    %1 = util.optimization_barrier %cst_0 : tensor<3xi8>
    %2 = tensor.empty() : tensor<9xi8>
    %3 = linalg.fill ins(%c0_i8 : i8) outs(%2 : tensor<9xi8>) -> tensor<9xi8>
    %4 = linalg.conv_1d ins(%0, %1 : tensor<11xi8>, tensor<3xi8>) outs(%3 : tensor<9xi8>) -> tensor<9xi8>
    %5 = util.optimization_barrier %cst : tensor<9xi8>
    "check.expect_eq"(%4, %5) : (tensor<9xi8>, tensor<9xi8>) -> ()
    return
  }
}
