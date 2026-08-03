module {
  func.func @"gather_rank1_f32_10-8"() {
    %cst = arith.constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00]> : tensor<8xf32>
    %cst_0 = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi32>
    %cst_1 = arith.constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00]> : tensor<10xf32>
    %0 = util.optimization_barrier %cst_1 : tensor<10xf32>
    %1 = util.optimization_barrier %cst_0 : tensor<8xi32>
    %2 = "stablehlo.gather"(%0, %1) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1>}> : (tensor<10xf32>, tensor<8xi32>) -> tensor<8xf32>
    %3 = util.optimization_barrier %cst : tensor<8xf32>
    "check.expect_almost_eq"(%2, %3) {rtol = 9.99999997E-7 : f32} : (tensor<8xf32>, tensor<8xf32>) -> ()
    return
  }
}
