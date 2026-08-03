module {
  func.func @"batch_matvec_i8_2_4_8-2_8"() {
    %cst = arith.constant dense<[[-116, 108, 76, 44], [-20, -52, -84, -116]]> : tensor<2x4xi8>
    %cst_0 = arith.constant dense<[[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]]> : tensor<2x8xi8>
    %c0_i8 = arith.constant 0 : i8
    %cst_1 = arith.constant dense<[[[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15], [16, 17, 18, 19, 20, 21, 22, 23], [24, 25, 26, 27, 28, 29, 30, 31]], [[32, 33, 34, 35, 36, 37, 38, 39], [40, 41, 42, 43, 44, 45, 46, 47], [48, 49, 50, 51, 52, 53, 54, 55], [56, 57, 58, 59, 60, 61, 62, 63]]]> : tensor<2x4x8xi8>
    %0 = util.optimization_barrier %cst_1 : tensor<2x4x8xi8>
    %1 = util.optimization_barrier %cst_0 : tensor<2x8xi8>
    %2 = tensor.empty() : tensor<2x4xi8>
    %3 = linalg.fill ins(%c0_i8 : i8) outs(%2 : tensor<2x4xi8>) -> tensor<2x4xi8>
    %4 = linalg.batch_matvec ins(%0, %1 : tensor<2x4x8xi8>, tensor<2x8xi8>) outs(%3 : tensor<2x4xi8>) -> tensor<2x4xi8>
    %5 = util.optimization_barrier %cst : tensor<2x4xi8>
    "check.expect_eq"(%4, %5) : (tensor<2x4xi8>, tensor<2x4xi8>) -> ()
    return
  }
}
