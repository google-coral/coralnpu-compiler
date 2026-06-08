func.func @main(%arg0: tensor<?x?xi32>) -> tensor<?xi32> {
  %c0 = stablehlo.constant dense<0> : tensor<i32>
  %0 = stablehlo.reduce(%arg0 init: %c0) applies stablehlo.add across dimensions = [1] : (tensor<?x?xi32>, tensor<i32>) -> tensor<?xi32>
  return %0 : tensor<?xi32>
}
