func.func @test_add_i32() {
  %arg0 = util.unfoldable_constant dense<[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]> : tensor<16xi32>
  %arg1 = util.unfoldable_constant dense<[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]> : tensor<16xi32>
  %res = stablehlo.add %arg0, %arg1 : tensor<16xi32>
  %expected = util.unfoldable_constant dense<[11, 22, 33, 44, 55, 66, 77, 88, 99, 110, 121, 132, 143, 154, 165, 176]> : tensor<16xi32>
  check.expect_eq(%res, %expected) : tensor<16xi32>
  return
}
