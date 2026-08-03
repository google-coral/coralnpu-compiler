module {
  func.func @reduce_sum_rank4_i8_2_2_3_2() {
    %cst = arith.constant dense<[[[1, 5, 9], [13, 17, 21]], [[25, 29, 33], [37, 41, 45]]]> : tensor<2x2x3xi8>
    %c = stablehlo.constant dense<0> : tensor<i8>
    %cst_0 = arith.constant dense<[[[[0, 1], [2, 3], [4, 5]], [[6, 7], [8, 9], [10, 11]]], [[[12, 13], [14, 15], [16, 17]], [[18, 19], [20, 21], [22, 23]]]]> : tensor<2x2x3x2xi8>
    %0 = util.optimization_barrier %cst_0 : tensor<2x2x3x2xi8>
    %1 = stablehlo.reduce(%0 init: %c) applies stablehlo.add across dimensions = [3] : (tensor<2x2x3x2xi8>, tensor<i8>) -> tensor<2x2x3xi8>
    %2 = util.optimization_barrier %cst : tensor<2x2x3xi8>
    "check.expect_eq"(%1, %2) : (tensor<2x2x3xi8>, tensor<2x2x3xi8>) -> ()
    return
  }
}
