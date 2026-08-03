module {
  func.func @"tuple_rank1_f32_8-8"() {
    %cst = arith.constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00]> : tensor<8xf32>
    %0 = util.optimization_barrier %cst : tensor<8xf32>
    %1 = util.optimization_barrier %cst : tensor<8xf32>
    %2 = stablehlo.tuple %0, %1 : tuple<tensor<8xf32>, tensor<8xf32>>
    %3 = stablehlo.get_tuple_element %2[0] : (tuple<tensor<8xf32>, tensor<8xf32>>) -> tensor<8xf32>
    %4 = stablehlo.get_tuple_element %2[1] : (tuple<tensor<8xf32>, tensor<8xf32>>) -> tensor<8xf32>
    %5 = util.optimization_barrier %cst : tensor<8xf32>
    "check.expect_almost_eq"(%3, %5) {rtol = 9.99999997E-7 : f32} : (tensor<8xf32>, tensor<8xf32>) -> ()
    %6 = util.optimization_barrier %cst : tensor<8xf32>
    "check.expect_almost_eq"(%4, %6) {rtol = 9.99999997E-7 : f32} : (tensor<8xf32>, tensor<8xf32>) -> ()
    return
  }
}
