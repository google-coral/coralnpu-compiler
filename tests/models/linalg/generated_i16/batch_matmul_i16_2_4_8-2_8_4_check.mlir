module {
  func.func @"batch_matmul_i16_2_4_8-2_8_4"() {
    %cst = arith.constant dense<[[[560, 588, 616, 644], [1456, 1548, 1640, 1732], [2352, 2508, 2664, 2820], [3248, 3468, 3688, 3908]], [[13232, 13516, 13800, 14084], [16176, 16524, 16872, 17220], [19120, 19532, 19944, 20356], [22064, 22540, 23016, 23492]]]> : tensor<2x4x4xi16>
    %cst_0 = arith.constant dense<[[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23], [24, 25, 26, 27], [28, 29, 30, 31]], [[32, 33, 34, 35], [36, 37, 38, 39], [40, 41, 42, 43], [44, 45, 46, 47], [48, 49, 50, 51], [52, 53, 54, 55], [56, 57, 58, 59], [60, 61, 62, 63]]]> : tensor<2x8x4xi16>
    %c0_i16 = arith.constant 0 : i16
    %cst_1 = arith.constant dense<[[[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15], [16, 17, 18, 19, 20, 21, 22, 23], [24, 25, 26, 27, 28, 29, 30, 31]], [[32, 33, 34, 35, 36, 37, 38, 39], [40, 41, 42, 43, 44, 45, 46, 47], [48, 49, 50, 51, 52, 53, 54, 55], [56, 57, 58, 59, 60, 61, 62, 63]]]> : tensor<2x4x8xi16>
    %0 = util.optimization_barrier %cst_1 : tensor<2x4x8xi16>
    %1 = util.optimization_barrier %cst_0 : tensor<2x8x4xi16>
    %2 = tensor.empty() : tensor<2x4x4xi16>
    %3 = linalg.fill ins(%c0_i16 : i16) outs(%2 : tensor<2x4x4xi16>) -> tensor<2x4x4xi16>
    %4 = linalg.batch_matmul ins(%0, %1 : tensor<2x4x8xi16>, tensor<2x8x4xi16>) outs(%3 : tensor<2x4x4xi16>) -> tensor<2x4x4xi16>
    %5 = util.optimization_barrier %cst : tensor<2x4x4xi16>
    "check.expect_eq"(%4, %5) : (tensor<2x4x4xi16>, tensor<2x4x4xi16>) -> ()
    return
  }
}
