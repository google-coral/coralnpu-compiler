func.func @main(%pred: tensor<?x?xi1>, %on_true: tensor<?x?xi16>, %on_false: tensor<?x?xi16>) -> tensor<?x?xi16> {
  %0 = stablehlo.select %pred, %on_true, %on_false : (tensor<?x?xi1>, tensor<?x?xi16>, tensor<?x?xi16>) -> tensor<?x?xi16>
  return %0 : tensor<?x?xi16>
}
