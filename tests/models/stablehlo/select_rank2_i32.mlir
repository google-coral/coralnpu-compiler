func.func @main(%pred: tensor<?x?xi1>, %on_true: tensor<?x?xi32>, %on_false: tensor<?x?xi32>) -> tensor<?x?xi32> {
  %0 = stablehlo.select %pred, %on_true, %on_false : (tensor<?x?xi1>, tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
  return %0 : tensor<?x?xi32>
}
