module {
  func.func @"mmt4d_i32_1_2_4_4-2_2_4_4"() {
    %cst = arith.constant dense<[[[[1244, 1548, 1852, 2156], [1548, 1980, 2412, 2844], [1852, 2412, 2972, 3532], [2156, 2844, 3532, 4220]], [[3676, 3980, 4284, 4588], [5004, 5436, 5868, 6300], [6332, 6892, 7452, 8012], [7660, 8348, 9036, 9724]]]]> : tensor<1x2x4x4xi32>
    %cst_0 = arith.constant dense<[[[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]], [[16, 17, 18, 19], [20, 21, 22, 23], [24, 25, 26, 27], [28, 29, 30, 31]]], [[[32, 33, 34, 35], [36, 37, 38, 39], [40, 41, 42, 43], [44, 45, 46, 47]], [[48, 49, 50, 51], [52, 53, 54, 55], [56, 57, 58, 59], [60, 61, 62, 63]]]]> : tensor<2x2x4x4xi32>
    %c0_i32 = arith.constant 0 : i32
    %cst_1 = arith.constant dense<[[[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]], [[16, 17, 18, 19], [20, 21, 22, 23], [24, 25, 26, 27], [28, 29, 30, 31]]]]> : tensor<1x2x4x4xi32>
    %0 = util.optimization_barrier %cst_1 : tensor<1x2x4x4xi32>
    %1 = util.optimization_barrier %cst_0 : tensor<2x2x4x4xi32>
    %2 = tensor.empty() : tensor<1x2x4x4xi32>
    %3 = linalg.fill ins(%c0_i32 : i32) outs(%2 : tensor<1x2x4x4xi32>) -> tensor<1x2x4x4xi32>
    %4 = linalg.mmt4d ins(%0, %1 : tensor<1x2x4x4xi32>, tensor<2x2x4x4xi32>) outs(%3 : tensor<1x2x4x4xi32>) -> tensor<1x2x4x4xi32>
    %5 = util.optimization_barrier %cst : tensor<1x2x4x4xi32>
    "check.expect_eq"(%4, %5) : (tensor<1x2x4x4xi32>, tensor<1x2x4x4xi32>) -> ()
    return
  }
}
