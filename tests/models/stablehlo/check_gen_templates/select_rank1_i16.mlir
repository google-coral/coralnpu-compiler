func.func @main(%pred: tensor<?xi1>, %on_true: tensor<?xi16>, %on_false: tensor<?xi16>) -> tensor<?xi16> {
  %0 = stablehlo.select %pred, %on_true, %on_false : (tensor<?xi1>, tensor<?xi16>, tensor<?xi16>) -> tensor<?xi16>
  return %0 : tensor<?xi16>
}
