module {
  func.func @broadcast_in_dim_rank3_i8_4_8_4_8() {
    %cst = arith.constant dense<"0x0001020304050607000102030405060700010203040506070001020304050607000102030405060708090A0B0C0D0E0F08090A0B0C0D0E0F08090A0B0C0D0E0F08090A0B0C0D0E0F08090A0B0C0D0E0F1011121314151617101112131415161710111213141516171011121314151617101112131415161718191A1B1C1D1E1F18191A1B1C1D1E1F18191A1B1C1D1E1F18191A1B1C1D1E1F18191A1B1C1D1E1F"> : tensor<4x5x8xi8>
    %cst_0 = arith.constant dense<[[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15], [16, 17, 18, 19, 20, 21, 22, 23], [24, 25, 26, 27, 28, 29, 30, 31]]> : tensor<4x8xi8>
    %0 = util.optimization_barrier %cst_0 : tensor<4x8xi8>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 2] : (tensor<4x8xi8>) -> tensor<4x5x8xi8>
    %2 = util.optimization_barrier %cst : tensor<4x5x8xi8>
    "check.expect_eq"(%1, %2) : (tensor<4x5x8xi8>, tensor<4x5x8xi8>) -> ()
    return
  }
}
