module {
  func.func @reduce_2d_max_i32_4_8() {
    %cst = arith.constant dense<[7, 15, 23, 31]> : tensor<4xi32>
    %c-2147483648_i32 = arith.constant -2147483648 : i32
    %cst_0 = arith.constant dense<[[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15], [16, 17, 18, 19, 20, 21, 22, 23], [24, 25, 26, 27, 28, 29, 30, 31]]> : tensor<4x8xi32>
    %0 = util.optimization_barrier %cst_0 : tensor<4x8xi32>
    %1 = tensor.empty() : tensor<4xi32>
    %2 = linalg.fill ins(%c-2147483648_i32 : i32) outs(%1 : tensor<4xi32>) -> tensor<4xi32>
    %reduced = linalg.reduce ins(%0 : tensor<4x8xi32>) outs(%2 : tensor<4xi32>) dimensions = [1] 
      (%in: i32, %init: i32) {
        %4 = arith.maxsi %in, %init : i32
        linalg.yield %4 : i32
      }
    %3 = util.optimization_barrier %cst : tensor<4xi32>
    "check.expect_eq"(%reduced, %3) : (tensor<4xi32>, tensor<4xi32>) -> ()
    return
  }
}
