module {
  func.func @reduce_2d_max_i8_4_8() {
    %cst = arith.constant dense<[7, 15, 23, 31]> : tensor<4xi8>
    %c-128_i8 = arith.constant -128 : i8
    %cst_0 = arith.constant dense<[[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15], [16, 17, 18, 19, 20, 21, 22, 23], [24, 25, 26, 27, 28, 29, 30, 31]]> : tensor<4x8xi8>
    %0 = util.optimization_barrier %cst_0 : tensor<4x8xi8>
    %1 = tensor.empty() : tensor<4xi8>
    %2 = linalg.fill ins(%c-128_i8 : i8) outs(%1 : tensor<4xi8>) -> tensor<4xi8>
    %reduced = linalg.reduce ins(%0 : tensor<4x8xi8>) outs(%2 : tensor<4xi8>) dimensions = [1] 
      (%in: i8, %init: i8) {
        %4 = arith.maxsi %in, %init : i8
        linalg.yield %4 : i8
      }
    %3 = util.optimization_barrier %cst : tensor<4xi8>
    "check.expect_eq"(%reduced, %3) : (tensor<4xi8>, tensor<4xi8>) -> ()
    return
  }
}
