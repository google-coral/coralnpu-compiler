module {
  func.func @reduce_2d_max_i16_4_8() {
    %cst = arith.constant dense<[7, 15, 23, 31]> : tensor<4xi16>
    %c-32768_i16 = arith.constant -32768 : i16
    %cst_0 = arith.constant dense<[[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15], [16, 17, 18, 19, 20, 21, 22, 23], [24, 25, 26, 27, 28, 29, 30, 31]]> : tensor<4x8xi16>
    %0 = util.optimization_barrier %cst_0 : tensor<4x8xi16>
    %1 = tensor.empty() : tensor<4xi16>
    %2 = linalg.fill ins(%c-32768_i16 : i16) outs(%1 : tensor<4xi16>) -> tensor<4xi16>
    %reduced = linalg.reduce ins(%0 : tensor<4x8xi16>) outs(%2 : tensor<4xi16>) dimensions = [1] 
      (%in: i16, %init: i16) {
        %4 = arith.maxsi %in, %init : i16
        linalg.yield %4 : i16
      }
    %3 = util.optimization_barrier %cst : tensor<4xi16>
    "check.expect_eq"(%reduced, %3) : (tensor<4xi16>, tensor<4xi16>) -> ()
    return
  }
}
