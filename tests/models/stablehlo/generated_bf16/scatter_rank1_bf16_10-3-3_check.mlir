module {
  func.func @"scatter_rank1_bf16_10-3-3"() {
    %cst = arith.constant dense<[0.000000e+00, 2.000000e+00, 4.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00]> : tensor<10xbf16>
    %cst_0 = arith.constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00]> : tensor<3xbf16>
    %cst_1 = arith.constant dense<[0, 1, 2]> : tensor<3xi32>
    %cst_2 = arith.constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00]> : tensor<10xbf16>
    %0 = util.optimization_barrier %cst_2 : tensor<10xbf16>
    %1 = util.optimization_barrier %cst_1 : tensor<3xi32>
    %2 = util.optimization_barrier %cst_0 : tensor<3xbf16>
    %3 = "stablehlo.scatter"(%0, %1, %2) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>}> ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %5 = stablehlo.add %arg0, %arg1 : tensor<bf16>
      stablehlo.return %5 : tensor<bf16>
    }) : (tensor<10xbf16>, tensor<3xi32>, tensor<3xbf16>) -> tensor<10xbf16>
    %4 = util.optimization_barrier %cst : tensor<10xbf16>
    "check.expect_almost_eq"(%3, %4) {atol = 0.00999999977 : f32, rtol = 0.00999999977 : f32} : (tensor<10xbf16>, tensor<10xbf16>) -> ()
    return
  }
}
