module {
  func.func @bitcast_convert_rank1_bf16_i16_8() {
    %cst = arith.constant dense<[0, 16256, 16384, 16448, 16512, 16544, 16576, 16608]> : tensor<8xi16>
    %cst_0 = arith.constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00]> : tensor<8xbf16>
    %0 = util.optimization_barrier %cst_0 : tensor<8xbf16>
    %1 = stablehlo.bitcast_convert %0 : (tensor<8xbf16>) -> tensor<8xi16>
    %2 = util.optimization_barrier %cst : tensor<8xi16>
    "check.expect_eq"(%1, %2) : (tensor<8xi16>, tensor<8xi16>) -> ()
    return
  }
}
