func.func @main(%pred: tensor<?x?xi1>, %on_true: tensor<?x?xf32>, %on_false: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = stablehlo.select %pred, %on_true, %on_false : (tensor<?x?xi1>, tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
