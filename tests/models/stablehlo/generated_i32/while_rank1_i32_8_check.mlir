module {
  func.func @while_rank1_i32_8() {
    %cst = arith.constant dense<[0, 32, 64, 96, 128, 160, 192, 224]> : tensor<8xi32>
    %c = stablehlo.constant dense<5> : tensor<i32>
    %c_0 = stablehlo.constant dense<0> : tensor<i32>
    %c_1 = stablehlo.constant dense<1> : tensor<i32>
    %cst_2 = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi32>
    %0 = util.optimization_barrier %cst_2 : tensor<8xi32>
    %1:2 = stablehlo.while(%iterArg = %0, %iterArg_3 = %c_0) : tensor<8xi32>, tensor<i32>
    cond {
      %3 = stablehlo.compare LT, %iterArg_3, %c : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %3 : tensor<i1>
    } do {
      %3 = stablehlo.add %iterArg_3, %c_1 : tensor<i32>
      %4 = stablehlo.add %iterArg, %iterArg : tensor<8xi32>
      stablehlo.return %4, %3 : tensor<8xi32>, tensor<i32>
    }
    %2 = util.optimization_barrier %cst : tensor<8xi32>
    "check.expect_eq"(%1#0, %2) : (tensor<8xi32>, tensor<8xi32>) -> ()
    return
  }
}
