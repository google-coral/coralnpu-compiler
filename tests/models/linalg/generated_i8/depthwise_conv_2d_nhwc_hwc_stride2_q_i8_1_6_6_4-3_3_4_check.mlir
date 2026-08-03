module {
  func.func @"depthwise_conv_2d_nhwc_hwc_stride2_q_i8_1_6_6_4-3_3_4"() {
    %cst = arith.constant dense<[[[[4848, 5190, 5550, 5928], [6360, 6774, 7206, 7656]], [[13920, 14694, 15486, 16296], [15432, 16278, 17142, 18024]]]]> : tensor<1x2x2x4xi32>
    %cst_0 = arith.constant dense<[[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]], [[24, 25, 26, 27], [28, 29, 30, 31], [32, 33, 34, 35]]]> : tensor<3x3x4xi8>
    %c0_i32 = arith.constant 0 : i32
    %c-5_i32 = arith.constant -5 : i32
    %c12_i32 = arith.constant 12 : i32
    %cst_1 = arith.constant dense<"0x000102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E1F202122232425262728292A2B2C2D2E2F303132333435363738393A3B3C3D3E3F404142434445464748494A4B4C4D4E4F505152535455565758595A5B5C5D5E5F606162636465666768696A6B6C6D6E6F707172737475767778797A7B7C7D7E7F808182838485868788898A8B8C8D8E8F"> : tensor<1x6x6x4xi8>
    %0 = util.optimization_barrier %cst_1 : tensor<1x6x6x4xi8>
    %1 = util.optimization_barrier %cst_0 : tensor<3x3x4xi8>
    %2 = tensor.empty() : tensor<1x2x2x4xi32>
    %3 = linalg.fill ins(%c0_i32 : i32) outs(%2 : tensor<1x2x2x4xi32>) -> tensor<1x2x2x4xi32>
    %4 = linalg.depthwise_conv_2d_nhwc_hwc_q {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%0, %1, %c12_i32, %c-5_i32 : tensor<1x6x6x4xi8>, tensor<3x3x4xi8>, i32, i32) outs(%3 : tensor<1x2x2x4xi32>) -> tensor<1x2x2x4xi32>
    %5 = util.optimization_barrier %cst : tensor<1x2x2x4xi32>
    "check.expect_eq"(%4, %5) : (tensor<1x2x2x4xi32>, tensor<1x2x2x4xi32>) -> ()
    return
  }
}
