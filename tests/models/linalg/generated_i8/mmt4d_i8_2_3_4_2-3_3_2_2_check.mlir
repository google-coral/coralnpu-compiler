module {
  func.func @"mmt4d_i8_2_3_4_2-3_3_2_2"() {
    %cst = arith.constant dense<[[[[103, -51], [-99, 27], [-45, 105], [9, -73]], [[-53, 49], [-111, 15], [87, -19], [29, -53]], [[47, -107], [-123, 3], [-37, 113], [49, -33]]], [[[-17, 117], [37, -61], [91, 17], [-111, 95]], [[19, -103], [-39, 119], [-97, 85], [101, 51]], [[55, -67], [-115, 43], [-29, -103], [57, 7]]]]> : tensor<2x3x4x2xi8>
    %cst_0 = arith.constant dense<[[[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]], [[[12, 13], [14, 15]], [[16, 17], [18, 19]], [[20, 21], [22, 23]]], [[[24, 25], [26, 27]], [[28, 29], [30, 31]], [[32, 33], [34, 35]]]]> : tensor<3x3x2x2xi8>
    %c0_i8 = arith.constant 0 : i8
    %cst_1 = arith.constant dense<[[[[0, 1], [2, 3], [4, 5], [6, 7]], [[8, 9], [10, 11], [12, 13], [14, 15]], [[16, 17], [18, 19], [20, 21], [22, 23]]], [[[24, 25], [26, 27], [28, 29], [30, 31]], [[32, 33], [34, 35], [36, 37], [38, 39]], [[40, 41], [42, 43], [44, 45], [46, 47]]]]> : tensor<2x3x4x2xi8>
    %0 = util.optimization_barrier %cst_1 : tensor<2x3x4x2xi8>
    %1 = util.optimization_barrier %cst_0 : tensor<3x3x2x2xi8>
    %2 = tensor.empty() : tensor<2x3x4x2xi8>
    %3 = linalg.fill ins(%c0_i8 : i8) outs(%2 : tensor<2x3x4x2xi8>) -> tensor<2x3x4x2xi8>
    %4 = linalg.mmt4d ins(%0, %1 : tensor<2x3x4x2xi8>, tensor<3x3x2x2xi8>) outs(%3 : tensor<2x3x4x2xi8>) -> tensor<2x3x4x2xi8>
    %5 = util.optimization_barrier %cst : tensor<2x3x4x2xi8>
    "check.expect_eq"(%4, %5) : (tensor<2x3x4x2xi8>, tensor<2x3x4x2xi8>) -> ()
    return
  }
}
