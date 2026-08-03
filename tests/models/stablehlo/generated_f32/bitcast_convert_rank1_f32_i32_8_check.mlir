module {
  func.func @bitcast_convert_rank1_f32_i32_8() {
    %cst = arith.constant dense<[0, 1065353216, 1073741824, 1077936128, 1082130432, 1084227584, 1086324736, 1088421888]> : tensor<8xi32>
    %cst_0 = arith.constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00]> : tensor<8xf32>
    %0 = util.optimization_barrier %cst_0 : tensor<8xf32>
    %1 = stablehlo.bitcast_convert %0 : (tensor<8xf32>) -> tensor<8xi32>
    %2 = util.optimization_barrier %cst : tensor<8xi32>
    "check.expect_eq"(%1, %2) : (tensor<8xi32>, tensor<8xi32>) -> ()
    return
  }
}
