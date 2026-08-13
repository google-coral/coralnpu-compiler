module {
  func.func @"conv_2d_nhwc_hwcf_dilation2_i8_1_6_6_4-3_3_4_4"() {
    %cst = arith.constant dense<[[[[-72, -50, -28, -6], [24, -66, 100, 10]], [[-8, 110, -28, 90], [88, 94, 100, 106]]]]> : tensor<1x2x2x4xi8>
    %cst_0 = arith.constant dense<"0x000102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E1F202122232425262728292A2B2C2D2E2F303132333435363738393A3B3C3D3E3F404142434445464748494A4B4C4D4E4F505152535455565758595A5B5C5D5E5F606162636465666768696A6B6C6D6E6F707172737475767778797A7B7C7D7E7F808182838485868788898A8B8C8D8E8F"> : tensor<3x3x4x4xi8>
    %c0_i8 = arith.constant 0 : i8
    %cst_1 = arith.constant dense<"0x000102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E1F202122232425262728292A2B2C2D2E2F303132333435363738393A3B3C3D3E3F404142434445464748494A4B4C4D4E4F505152535455565758595A5B5C5D5E5F606162636465666768696A6B6C6D6E6F707172737475767778797A7B7C7D7E7F808182838485868788898A8B8C8D8E8F"> : tensor<1x6x6x4xi8>
    %0 = util.optimization_barrier %cst_1 : tensor<1x6x6x4xi8>
    %1 = util.optimization_barrier %cst_0 : tensor<3x3x4x4xi8>
    %2 = tensor.empty() : tensor<1x2x2x4xi8>
    %3 = linalg.fill ins(%c0_i8 : i8) outs(%2 : tensor<1x2x2x4xi8>) -> tensor<1x2x2x4xi8>
    %4 = linalg.conv_2d_nhwc_hwcf {dilations = dense<2> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%0, %1 : tensor<1x6x6x4xi8>, tensor<3x3x4x4xi8>) outs(%3 : tensor<1x2x2x4xi8>) -> tensor<1x2x2x4xi8>
    %5 = util.optimization_barrier %cst : tensor<1x2x2x4xi8>
    "check.expect_eq"(%4, %5) : (tensor<1x2x2x4xi8>, tensor<1x2x2x4xi8>) -> ()
    return
  }
}
