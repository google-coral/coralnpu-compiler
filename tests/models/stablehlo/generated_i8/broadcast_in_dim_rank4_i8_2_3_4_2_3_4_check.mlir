module {
  func.func @broadcast_in_dim_rank4_i8_2_3_4_2_3_4() {
    %cst = arith.constant dense<"0x00010203000102030001020300010203000102030001020304050607040506070405060704050607040506070405060708090A0B08090A0B08090A0B08090A0B08090A0B08090A0B0C0D0E0F0C0D0E0F0C0D0E0F0C0D0E0F0C0D0E0F0C0D0E0F101112131011121310111213101112131011121310111213141516171415161714151617141516171415161714151617"> : tensor<2x3x6x4xi8>
    %cst_0 = arith.constant dense<[[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]> : tensor<2x3x4xi8>
    %0 = util.optimization_barrier %cst_0 : tensor<2x3x4xi8>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 3] : (tensor<2x3x4xi8>) -> tensor<2x3x6x4xi8>
    %2 = util.optimization_barrier %cst : tensor<2x3x6x4xi8>
    "check.expect_eq"(%1, %2) : (tensor<2x3x6x4xi8>, tensor<2x3x6x4xi8>) -> ()
    return
  }
}
