module {
  func.func @fill_rng_2d_i8_4_8() {
    %cst = arith.constant dense<[[1.900000e+01, 5.100000e+01, 8.400000e+01, 1.170000e+02, 2.200000e+01, 5.500000e+01, 8.800000e+01, 1.200000e+02], [1.150000e+02, 2.100000e+01, 5.400000e+01, 8.600000e+01, 1.190000e+02, 2.500000e+01, 5.700000e+01, 9.000000e+01], [8.500000e+01, 1.180000e+02, 2.300000e+01, 5.600000e+01, 8.800000e+01, 1.210000e+02, 2.700000e+01, 5.900000e+01], [5.500000e+01, 8.700000e+01, 1.200000e+02, 2.500000e+01, 5.800000e+01, 9.100000e+01, 1.230000e+02, 2.900000e+01]]> : tensor<4x8xf32>
    %c12345_i32 = arith.constant 12345 : i32
    %cst_0 = arith.constant 1.270000e+02 : f64
    %cst_1 = arith.constant 0.000000e+00 : f64
    %cst_2 = arith.constant 2.3283063999999999E-10 : f64
    %cst_3 = arith.constant 0x41DFFFFFFFC00000 : f64
    %c1103515245_i32 = arith.constant 1103515245 : i32
    %cst_4 = arith.constant dense<[[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15], [16, 17, 18, 19, 20, 21, 22, 23], [24, 25, 26, 27, 28, 29, 30, 31]]> : tensor<4x8xi8>
    %0 = util.optimization_barrier %cst_4 : tensor<4x8xi8>
    %1 = util.optimization_barrier %c12345_i32 : i32
    %2 = tensor.empty() : tensor<4x8xi8>
    %3 = linalg.fill_rng_2d ins(%cst_1, %cst_0, %1 : f64, f64, i32) outs(%2 : tensor<4x8xi8>) -> tensor<4x8xi8>
    %cast = tensor.cast %3 : tensor<4x8xi8> to tensor<?x?xi8>
    %4 = arith.sitofp %cast : tensor<?x?xi8> to tensor<?x?xf32>
    %cast_5 = tensor.cast %4 : tensor<?x?xf32> to tensor<4x8xf32>
    %5 = util.optimization_barrier %cst : tensor<4x8xf32>
    check.expect_almost_eq(%cast_5, %5, atol 1.000000e+00, rtol 9.99999997E-7) : tensor<4x8xf32>
    return
  }
}
