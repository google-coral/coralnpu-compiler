func.func @main(%pred: tensor<?xi1>, %on_true: tensor<?xf32>, %on_false: tensor<?xf32>) -> tensor<?xf32> {
  %0 = stablehlo.select %pred, %on_true, %on_false : (tensor<?xi1>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}
