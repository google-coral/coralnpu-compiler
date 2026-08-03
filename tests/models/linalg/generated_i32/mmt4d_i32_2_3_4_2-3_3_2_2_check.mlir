module {
  func.func @"mmt4d_i32_2_3_4_2-3_3_2_2"() {
    %cst = arith.constant dense<[[[[359, 461], [413, 539], [467, 617], [521, 695]], [[971, 1073], [1169, 1295], [1367, 1517], [1565, 1739]], [[1583, 1685], [1925, 2051], [2267, 2417], [2609, 2783]]], [[[1007, 1397], [1061, 1475], [1115, 1553], [1169, 1631]], [[3347, 3737], [3545, 3959], [3743, 4181], [3941, 4403]], [[5687, 6077], [6029, 6443], [6371, 6809], [6713, 7175]]]]> : tensor<2x3x4x2xi32>
    %cst_0 = arith.constant dense<[[[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]], [[[12, 13], [14, 15]], [[16, 17], [18, 19]], [[20, 21], [22, 23]]], [[[24, 25], [26, 27]], [[28, 29], [30, 31]], [[32, 33], [34, 35]]]]> : tensor<3x3x2x2xi32>
    %c0_i32 = arith.constant 0 : i32
    %cst_1 = arith.constant dense<[[[[0, 1], [2, 3], [4, 5], [6, 7]], [[8, 9], [10, 11], [12, 13], [14, 15]], [[16, 17], [18, 19], [20, 21], [22, 23]]], [[[24, 25], [26, 27], [28, 29], [30, 31]], [[32, 33], [34, 35], [36, 37], [38, 39]], [[40, 41], [42, 43], [44, 45], [46, 47]]]]> : tensor<2x3x4x2xi32>
    %0 = util.optimization_barrier %cst_1 : tensor<2x3x4x2xi32>
    %1 = util.optimization_barrier %cst_0 : tensor<3x3x2x2xi32>
    %2 = tensor.empty() : tensor<2x3x4x2xi32>
    %3 = linalg.fill ins(%c0_i32 : i32) outs(%2 : tensor<2x3x4x2xi32>) -> tensor<2x3x4x2xi32>
    %4 = linalg.mmt4d ins(%0, %1 : tensor<2x3x4x2xi32>, tensor<3x3x2x2xi32>) outs(%3 : tensor<2x3x4x2xi32>) -> tensor<2x3x4x2xi32>
    %5 = util.optimization_barrier %cst : tensor<2x3x4x2xi32>
    "check.expect_eq"(%4, %5) : (tensor<2x3x4x2xi32>, tensor<2x3x4x2xi32>) -> ()
    return
  }
}
