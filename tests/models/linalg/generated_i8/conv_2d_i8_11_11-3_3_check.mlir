module {
  func.func @"conv_2d_i8_11_11-3_3"() {
    %cst = arith.constant dense<[[124, -96, -60, -24, 12, 48, 84, 120, -100], [8, 44, 80, 116, -104, -68, -32, 4, 40], [-108, -72, -36, 0, 36, 72, 108, -112, -76], [32, 68, 104, -116, -80, -44, -8, 28, 64], [-84, -48, -12, 24, 60, 96, -124, -88, -52], [56, 92, -128, -92, -56, -20, 16, 52, 88], [-60, -24, 12, 48, 84, 120, -100, -64, -28], [80, 116, -104, -68, -32, 4, 40, 76, 112], [-36, 0, 36, 72, 108, -112, -76, -40, -4]]> : tensor<9x9xi8>
    %cst_0 = arith.constant dense<[[0, 1, 2], [3, 4, 5], [6, 7, 8]]> : tensor<3x3xi8>
    %c0_i8 = arith.constant 0 : i8
    %cst_1 = arith.constant dense<"0x000102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E1F202122232425262728292A2B2C2D2E2F303132333435363738393A3B3C3D3E3F404142434445464748494A4B4C4D4E4F505152535455565758595A5B5C5D5E5F606162636465666768696A6B6C6D6E6F707172737475767778"> : tensor<11x11xi8>
    %0 = util.optimization_barrier %cst_1 : tensor<11x11xi8>
    %1 = util.optimization_barrier %cst_0 : tensor<3x3xi8>
    %2 = tensor.empty() : tensor<9x9xi8>
    %3 = linalg.fill ins(%c0_i8 : i8) outs(%2 : tensor<9x9xi8>) -> tensor<9x9xi8>
    %4 = linalg.conv_2d ins(%0, %1 : tensor<11x11xi8>, tensor<3x3xi8>) outs(%3 : tensor<9x9xi8>) -> tensor<9x9xi8>
    %5 = util.optimization_barrier %cst : tensor<9x9xi8>
    "check.expect_eq"(%4, %5) : (tensor<9x9xi8>, tensor<9x9xi8>) -> ()
    return
  }
}
