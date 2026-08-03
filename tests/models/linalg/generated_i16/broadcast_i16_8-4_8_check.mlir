module {
  func.func @"broadcast_i16_8-4_8"() {
    %cst = arith.constant dense<[[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<4x8xi16>
    %cst_0 = arith.constant dense<[[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15], [16, 17, 18, 19, 20, 21, 22, 23], [24, 25, 26, 27, 28, 29, 30, 31]]> : tensor<4x8xi16>
    %cst_1 = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi16>
    %0 = util.optimization_barrier %cst_1 : tensor<8xi16>
    %1 = util.optimization_barrier %cst_0 : tensor<4x8xi16>
    %2 = tensor.empty() : tensor<4x8xi16>
    %broadcasted = linalg.broadcast ins(%0 : tensor<8xi16>) outs(%2 : tensor<4x8xi16>) dimensions = [0] 
    %3 = util.optimization_barrier %cst : tensor<4x8xi16>
    "check.expect_eq"(%broadcasted, %3) : (tensor<4x8xi16>, tensor<4x8xi16>) -> ()
    return
  }
}
