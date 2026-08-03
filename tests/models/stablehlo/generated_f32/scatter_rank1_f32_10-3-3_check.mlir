module {
  func.func @"scatter_rank1_f32_10-3-3"() {
    %cst = arith.constant dense<[0.000000e+00, 2.000000e+00, 4.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00]> : tensor<10xf32>
    %cst_0 = arith.constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00]> : tensor<3xf32>
    %cst_1 = arith.constant dense<[0, 1, 2]> : tensor<3xi32>
    %cst_2 = arith.constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00]> : tensor<10xf32>
    %0 = util.optimization_barrier %cst_2 : tensor<10xf32>
    %1 = util.optimization_barrier %cst_1 : tensor<3xi32>
    %2 = util.optimization_barrier %cst_0 : tensor<3xf32>
    %3 = "stablehlo.scatter"(%0, %1, %2) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>}> ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %5 = stablehlo.add %arg0, %arg1 : tensor<f32>
      stablehlo.return %5 : tensor<f32>
    }) : (tensor<10xf32>, tensor<3xi32>, tensor<3xf32>) -> tensor<10xf32>
    %4 = util.optimization_barrier %cst : tensor<10xf32>
    "check.expect_almost_eq"(%3, %4) {rtol = 9.99999997E-7 : f32} : (tensor<10xf32>, tensor<10xf32>) -> ()
    return
  }
}
