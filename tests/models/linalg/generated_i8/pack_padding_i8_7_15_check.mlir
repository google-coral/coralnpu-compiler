module {
  func.func @pack_padding_i8_7_15() {
    %cst = arith.constant dense<"0x000102030405060708090A0B0C0D0E000F101112131415161718191A1B1C1D001E1F202122232425262728292A2B2C002D2E2F303132333435363738393A3B003C3D3E3F404142434445464748494A004B4C4D4E4F50515253545556575859005A5B5C5D5E5F6061626364656667680000000000000000000000000000000000"> : tensor<1x1x8x16xi8>
    %c0_i8 = arith.constant 0 : i8
    %cst_0 = arith.constant dense<"0x000102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E1F202122232425262728292A2B2C2D2E2F303132333435363738393A3B3C3D3E3F404142434445464748494A4B4C4D4E4F505152535455565758595A5B5C5D5E5F606162636465666768"> : tensor<7x15xi8>
    %0 = util.optimization_barrier %cst_0 : tensor<7x15xi8>
    %1 = tensor.empty() : tensor<1x1x8x16xi8>
    %pack = linalg.pack %0 padding_value(%c0_i8 : i8) inner_dims_pos = [0, 1] inner_tiles = [8, 16] into %1 : tensor<7x15xi8> -> tensor<1x1x8x16xi8>
    %2 = util.optimization_barrier %cst : tensor<1x1x8x16xi8>
    "check.expect_eq"(%pack, %2) : (tensor<1x1x8x16xi8>, tensor<1x1x8x16xi8>) -> ()
    return
  }
}
