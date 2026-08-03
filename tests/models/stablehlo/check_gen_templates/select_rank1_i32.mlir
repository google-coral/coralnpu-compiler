func.func @main(%pred: tensor<?xi1>, %on_true: tensor<?xi32>, %on_false: tensor<?xi32>) -> tensor<?xi32> {
  %0 = stablehlo.select %pred, %on_true, %on_false : (tensor<?xi1>, tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
  return %0 : tensor<?xi32>
}
