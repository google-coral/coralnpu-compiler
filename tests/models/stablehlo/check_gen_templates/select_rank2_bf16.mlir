func.func @main(%pred: tensor<?x?xi1>, %on_true: tensor<?x?xbf16>, %on_false: tensor<?x?xbf16>) -> tensor<?x?xbf16> {
  %0 = stablehlo.select %pred, %on_true, %on_false : (tensor<?x?xi1>, tensor<?x?xbf16>, tensor<?x?xbf16>) -> tensor<?x?xbf16>
  return %0 : tensor<?x?xbf16>
}
