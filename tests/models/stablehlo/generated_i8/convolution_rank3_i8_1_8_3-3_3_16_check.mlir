module {
  func.func @"convolution_rank3_i8_1_8_3-3_3_16"() {
    %cst = arith.constant dense<[[[-64, -28, 8, 44, 80, 116, -104, -68, -32, 4, 40, 76, 112, -108, -72, -36], [-128, -65, -2, 61, 124, -69, -6, 57, 120, -73, -10, 53, 116, -77, -14, 49], [64, -102, -12, 78, -88, 2, 92, -74, 16, 106, -60, 30, 120, -46, 44, -122], [0, 117, -22, 95, -44, 73, -66, 51, -88, 29, -110, 7, 124, -15, 102, -37], [-64, 80, -32, 112, 0, -112, 32, -80, 64, -48, 96, -16, -128, 16, -96, 48], [-128, 43, -42, -127, 44, -41, -126, 45, -40, -125, 46, -39, -124, 47, -38, -123]]]> : tensor<1x6x16xi8>
    %cst_0 = arith.constant dense<"0x000102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E1F202122232425262728292A2B2C2D2E2F303132333435363738393A3B3C3D3E3F404142434445464748494A4B4C4D4E4F505152535455565758595A5B5C5D5E5F606162636465666768696A6B6C6D6E6F707172737475767778797A7B7C7D7E7F808182838485868788898A8B8C8D8E8F"> : tensor<3x3x16xi8>
    %cst_1 = arith.constant dense<[[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]]]> : tensor<1x8x3xi8>
    %0 = util.optimization_barrier %cst_1 : tensor<1x8x3xi8>
    %1 = util.optimization_barrier %cst_0 : tensor<3x3x16xi8>
    %2 = stablehlo.convolution(%0, %1) dim_numbers = [b, 0, f]x[0, i, o]->[b, 0, f], window = {stride = [1], pad = [[0, 0]], lhs_dilate = [1], rhs_dilate = [1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x8x3xi8>, tensor<3x3x16xi8>) -> tensor<1x6x16xi8>
    %3 = util.optimization_barrier %cst : tensor<1x6x16xi8>
    "check.expect_eq"(%2, %3) : (tensor<1x6x16xi8>, tensor<1x6x16xi8>) -> ()
    return
  }
}
