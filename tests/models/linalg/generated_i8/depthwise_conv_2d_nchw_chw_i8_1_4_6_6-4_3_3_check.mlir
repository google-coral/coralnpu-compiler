module {
  func.func @"depthwise_conv_2d_nchw_chw_i8_1_4_6_6-4_3_3"() {
    %cst = arith.constant dense<[[[[110, -110, -74, -38], [70, 106, -114, -78], [30, 66, 102, -118], [-10, 26, 62, 98]], [[25, -114, 3, 120], [-41, 76, -63, 54], [-107, 10, 127, -12], [83, -56, 61, -78]], [[-116, 82, 24, -34], [48, -10, -68, -126], [-44, -102, 96, 38], [120, 62, 4, -54]], [[-57, -34, -11, 12], [81, 104, 127, -106], [-37, -14, 9, 32], [101, 124, -109, -86]]]]> : tensor<1x4x4x4xi8>
    %cst_0 = arith.constant dense<[[[0, 1, 2], [3, 4, 5], [6, 7, 8]], [[9, 10, 11], [12, 13, 14], [15, 16, 17]], [[18, 19, 20], [21, 22, 23], [24, 25, 26]], [[27, 28, 29], [30, 31, 32], [33, 34, 35]]]> : tensor<4x3x3xi8>
    %c0_i8 = arith.constant 0 : i8
    %cst_1 = arith.constant dense<"0x000102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E1F202122232425262728292A2B2C2D2E2F303132333435363738393A3B3C3D3E3F404142434445464748494A4B4C4D4E4F505152535455565758595A5B5C5D5E5F606162636465666768696A6B6C6D6E6F707172737475767778797A7B7C7D7E7F808182838485868788898A8B8C8D8E8F"> : tensor<1x4x6x6xi8>
    %0 = util.optimization_barrier %cst_1 : tensor<1x4x6x6xi8>
    %1 = util.optimization_barrier %cst_0 : tensor<4x3x3xi8>
    %2 = tensor.empty() : tensor<1x4x4x4xi8>
    %3 = linalg.fill ins(%c0_i8 : i8) outs(%2 : tensor<1x4x4x4xi8>) -> tensor<1x4x4x4xi8>
    %4 = linalg.depthwise_conv_2d_nchw_chw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%0, %1 : tensor<1x4x6x6xi8>, tensor<4x3x3xi8>) outs(%3 : tensor<1x4x4x4xi8>) -> tensor<1x4x4x4xi8>
    %5 = util.optimization_barrier %cst : tensor<1x4x4x4xi8>
    "check.expect_eq"(%4, %5) : (tensor<1x4x4x4xi8>, tensor<1x4x4x4xi8>) -> ()
    return
  }
}
