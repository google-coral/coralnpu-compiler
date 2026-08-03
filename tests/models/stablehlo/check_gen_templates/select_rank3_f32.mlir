func.func @main(%pred: tensor<?x?x?xi1>, %on_true: tensor<?x?x?xf32>, %on_false: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %0 = stablehlo.select %pred, %on_true, %on_false : (tensor<?x?x?xi1>, tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
}
