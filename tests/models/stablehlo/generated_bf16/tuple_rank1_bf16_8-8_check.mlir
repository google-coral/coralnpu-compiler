module {
  func.func @"tuple_rank1_bf16_8-8"() {
    %cst = arith.constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00]> : tensor<8xbf16>
    %0 = util.optimization_barrier %cst : tensor<8xbf16>
    %1 = util.optimization_barrier %cst : tensor<8xbf16>
    %2 = stablehlo.tuple %0, %1 : tuple<tensor<8xbf16>, tensor<8xbf16>>
    %3 = stablehlo.get_tuple_element %2[0] : (tuple<tensor<8xbf16>, tensor<8xbf16>>) -> tensor<8xbf16>
    %4 = stablehlo.get_tuple_element %2[1] : (tuple<tensor<8xbf16>, tensor<8xbf16>>) -> tensor<8xbf16>
    %5 = util.optimization_barrier %cst : tensor<8xbf16>
    "check.expect_almost_eq"(%3, %5) {atol = 0.00999999977 : f32, rtol = 0.00999999977 : f32} : (tensor<8xbf16>, tensor<8xbf16>) -> ()
    %6 = util.optimization_barrier %cst : tensor<8xbf16>
    "check.expect_almost_eq"(%4, %6) {atol = 0.00999999977 : f32, rtol = 0.00999999977 : f32} : (tensor<8xbf16>, tensor<8xbf16>) -> ()
    return
  }
}
