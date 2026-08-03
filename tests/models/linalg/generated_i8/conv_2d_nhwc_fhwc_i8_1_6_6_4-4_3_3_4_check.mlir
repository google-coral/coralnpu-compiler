module {
  func.func @"conv_2d_nhwc_fhwc_i8_1_6_6_4-4_3_3_4"() {
    %cst = arith.constant dense<[[[[70, -98, -10, 78], [30, -74, 78, -26], [-10, -50, -90, 126], [-50, -26, -2, 22]], [[86, 46, 6, -34], [46, 70, 94, 118], [6, 94, -74, 14], [-34, 118, 14, -90]], [[102, -66, 22, 110], [62, -42, 110, 6], [22, -18, -58, -98], [-18, 6, 30, 54]], [[118, 78, 38, -2], [78, 102, 126, -106], [38, 126, -42, 46], [-2, -106, 46, -58]]]]> : tensor<1x4x4x4xi8>
    %cst_0 = arith.constant dense<"0x000102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E1F202122232425262728292A2B2C2D2E2F303132333435363738393A3B3C3D3E3F404142434445464748494A4B4C4D4E4F505152535455565758595A5B5C5D5E5F606162636465666768696A6B6C6D6E6F707172737475767778797A7B7C7D7E7F808182838485868788898A8B8C8D8E8F"> : tensor<4x3x3x4xi8>
    %c0_i8 = arith.constant 0 : i8
    %cst_1 = arith.constant dense<"0x000102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E1F202122232425262728292A2B2C2D2E2F303132333435363738393A3B3C3D3E3F404142434445464748494A4B4C4D4E4F505152535455565758595A5B5C5D5E5F606162636465666768696A6B6C6D6E6F707172737475767778797A7B7C7D7E7F808182838485868788898A8B8C8D8E8F"> : tensor<1x6x6x4xi8>
    %0 = util.optimization_barrier %cst_1 : tensor<1x6x6x4xi8>
    %1 = util.optimization_barrier %cst_0 : tensor<4x3x3x4xi8>
    %2 = tensor.empty() : tensor<1x4x4x4xi8>
    %3 = linalg.fill ins(%c0_i8 : i8) outs(%2 : tensor<1x4x4x4xi8>) -> tensor<1x4x4x4xi8>
    %4 = linalg.conv_2d_nhwc_fhwc {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%0, %1 : tensor<1x6x6x4xi8>, tensor<4x3x3x4xi8>) outs(%3 : tensor<1x4x4x4xi8>) -> tensor<1x4x4x4xi8>
    %5 = util.optimization_barrier %cst : tensor<1x4x4x4xi8>
    "check.expect_eq"(%4, %5) : (tensor<1x4x4x4xi8>, tensor<1x4x4x4xi8>) -> ()
    return
  }
}
