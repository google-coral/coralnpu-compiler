func.func @main(%arg0: tensor<?xi32>) -> tensor<?xi32> {
  %c_limit = stablehlo.constant dense<5> : tensor<i32>
  %c0 = stablehlo.constant dense<0> : tensor<i32>
  %0:2 = stablehlo.while(%iter = %arg0, %count = %c0) : tensor<?xi32>, tensor<i32>
    cond {
      %cond = "stablehlo.compare"(%count, %c_limit) {comparison_direction = #stablehlo<comparison_direction LT>} : (tensor<i32>, tensor<i32>) -> tensor<i1>
      "stablehlo.return"(%cond) : (tensor<i1>) -> ()
    } do {
      %c1 = stablehlo.constant dense<1> : tensor<i32>
      %next_count = stablehlo.add %count, %c1 : tensor<i32>
      %next_iter = stablehlo.add %iter, %iter : tensor<?xi32>
      "stablehlo.return"(%next_iter, %next_count) : (tensor<?xi32>, tensor<i32>) -> ()
    }
  return %0#0 : tensor<?xi32>
}
