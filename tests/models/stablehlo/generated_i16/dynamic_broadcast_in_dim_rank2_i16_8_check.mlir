module {
  func.func @dynamic_broadcast_in_dim_rank2_i16_8() {
    %cst = arith.constant dense<[[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<4x8xi16>
    %cst_0 = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi16>
    %0 = util.optimization_barrier %cst_0 : tensor<8xi16>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1] : (tensor<8xi16>) -> tensor<4x8xi16>
    %2 = util.optimization_barrier %cst : tensor<4x8xi16>
    "check.expect_eq"(%1, %2) : (tensor<4x8xi16>, tensor<4x8xi16>) -> ()
    return
  }
}
