module {
  func.func @slice_rank1_i8_8() {
    %cst = arith.constant dense<[2, 3, 4, 5]> : tensor<4xi8>
    %cst_0 = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi8>
    %0 = util.optimization_barrier %cst_0 : tensor<8xi8>
    %1 = stablehlo.slice %0 [2:6] : (tensor<8xi8>) -> tensor<4xi8>
    %2 = util.optimization_barrier %cst : tensor<4xi8>
    "check.expect_eq"(%1, %2) : (tensor<4xi8>, tensor<4xi8>) -> ()
    return
  }
}
