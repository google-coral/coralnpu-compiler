module {
  func.func @fill_rng_2d_bf16_4_8() {
    %cst = arith.constant dense<[[1.523440e-01, 4.101560e-01, 6.679690e-01, 9.218750e-01, 1.796880e-01, 4.375000e-01, 6.953130e-01, 9.492180e-01], [9.140620e-01, 1.699220e-01, 4.257810e-01, 6.835930e-01, 9.414060e-01, 1.972660e-01, 4.550780e-01, 7.109380e-01], [6.718750e-01, 9.296870e-01, 1.865230e-01, 4.433590e-01, 6.992180e-01, 9.570310e-01, 2.148440e-01, 4.707030e-01], [4.335940e-01, 6.914060e-01, 9.453120e-01, 2.041020e-01, 4.609380e-01, 7.187500e-01, 9.765620e-01, 2.314450e-01]]> : tensor<4x8xbf16>
    %c12345_i32 = arith.constant 12345 : i32
    %cst_0 = arith.constant 1.000000e+00 : f64
    %cst_1 = arith.constant 0.000000e+00 : f64
    %cst_2 = arith.constant 2.3283063999999999E-10 : f64
    %cst_3 = arith.constant 0x41DFFFFFFFC00000 : f64
    %c1103515245_i32 = arith.constant 1103515245 : i32
    %cst_4 = arith.constant dense<[[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00], [8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01], [1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01], [2.400000e+01, 2.500000e+01, 2.600000e+01, 2.700000e+01, 2.800000e+01, 2.900000e+01, 3.000000e+01, 3.100000e+01]]> : tensor<4x8xbf16>
    %0 = util.optimization_barrier %cst_4 : tensor<4x8xbf16>
    %1 = util.optimization_barrier %c12345_i32 : i32
    %2 = tensor.empty() : tensor<4x8xbf16>
    %3 = linalg.fill_rng_2d ins(%cst_1, %cst_0, %1 : f64, f64, i32) outs(%2 : tensor<4x8xbf16>) -> tensor<4x8xbf16>
    %4 = util.optimization_barrier %cst : tensor<4x8xbf16>
    "check.expect_almost_eq"(%3, %4) {atol = 0.00999999977 : f32, rtol = 0.00999999977 : f32} : (tensor<4x8xbf16>, tensor<4x8xbf16>) -> ()
    return
  }
}
