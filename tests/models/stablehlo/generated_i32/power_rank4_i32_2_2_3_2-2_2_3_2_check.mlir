module {
  func.func @"power_rank4_i32_2_2_3_2-2_2_3_2"() {
    %cst = arith.constant dense<[[[[1, 1], [4, 27], [256, 3125]], [[46656, 823543], [16777216, 387420489], [1410065408, 1843829075]]], [[[-251658240, -1692154371], [-1282129920, 1500973039], [0, 1681328401]], [[457441280, -306639989], [0, 878082373], [977272832, -1276351769]]]]> : tensor<2x2x3x2xi32>
    %cst_0 = arith.constant dense<[[[[0, 1], [2, 3], [4, 5]], [[6, 7], [8, 9], [10, 11]]], [[[12, 13], [14, 15], [16, 17]], [[18, 19], [20, 21], [22, 23]]]]> : tensor<2x2x3x2xi32>
    %0 = util.optimization_barrier %cst_0 : tensor<2x2x3x2xi32>
    %1 = util.optimization_barrier %cst_0 : tensor<2x2x3x2xi32>
    %2 = stablehlo.power %0, %1 : tensor<2x2x3x2xi32>
    %3 = util.optimization_barrier %cst : tensor<2x2x3x2xi32>
    "check.expect_eq"(%2, %3) : (tensor<2x2x3x2xi32>, tensor<2x2x3x2xi32>) -> ()
    return
  }
}
