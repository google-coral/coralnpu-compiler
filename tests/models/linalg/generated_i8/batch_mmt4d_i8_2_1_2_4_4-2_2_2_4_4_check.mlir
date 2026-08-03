module {
  func.func @"batch_mmt4d_i8_2_1_2_4_4-2_2_2_4_4"() {
    %cst = arith.constant dense<[[[[[-36, 12, 60, 108], [12, -68, 108, 28], [60, 108, -100, -52], [108, 28, -52, 124]], [[92, -116, -68, -20], [-116, 60, -20, -100], [-68, -20, 28, 76], [-20, -100, 76, -4]]]], [[[[92, -116, -68, -20], [-116, 60, -20, -100], [-68, -20, 28, 76], [-20, -100, 76, -4]], [[-36, 12, 60, 108], [12, -68, 108, 28], [60, 108, -100, -52], [108, 28, -52, 124]]]]]> : tensor<2x1x2x4x4xi8>
    %cst_0 = arith.constant dense<"0x000102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E1F202122232425262728292A2B2C2D2E2F303132333435363738393A3B3C3D3E3F404142434445464748494A4B4C4D4E4F505152535455565758595A5B5C5D5E5F606162636465666768696A6B6C6D6E6F707172737475767778797A7B7C7D7E7F"> : tensor<2x2x2x4x4xi8>
    %c0_i8 = arith.constant 0 : i8
    %cst_1 = arith.constant dense<[[[[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]], [[16, 17, 18, 19], [20, 21, 22, 23], [24, 25, 26, 27], [28, 29, 30, 31]]]], [[[[32, 33, 34, 35], [36, 37, 38, 39], [40, 41, 42, 43], [44, 45, 46, 47]], [[48, 49, 50, 51], [52, 53, 54, 55], [56, 57, 58, 59], [60, 61, 62, 63]]]]]> : tensor<2x1x2x4x4xi8>
    %0 = util.optimization_barrier %cst_1 : tensor<2x1x2x4x4xi8>
    %1 = util.optimization_barrier %cst_0 : tensor<2x2x2x4x4xi8>
    %2 = tensor.empty() : tensor<2x1x2x4x4xi8>
    %3 = linalg.fill ins(%c0_i8 : i8) outs(%2 : tensor<2x1x2x4x4xi8>) -> tensor<2x1x2x4x4xi8>
    %4 = linalg.batch_mmt4d ins(%0, %1 : tensor<2x1x2x4x4xi8>, tensor<2x2x2x4x4xi8>) outs(%3 : tensor<2x1x2x4x4xi8>) -> tensor<2x1x2x4x4xi8>
    %5 = util.optimization_barrier %cst : tensor<2x1x2x4x4xi8>
    "check.expect_eq"(%4, %5) : (tensor<2x1x2x4x4xi8>, tensor<2x1x2x4x4xi8>) -> ()
    return
  }
}
