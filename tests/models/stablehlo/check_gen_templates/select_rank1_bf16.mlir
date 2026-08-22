func.func @main(%pred: tensor<?xi1>, %on_true: tensor<?xbf16>, %on_false: tensor<?xbf16>) -> tensor<?xbf16> {
  %0 = stablehlo.select %pred, %on_true, %on_false : (tensor<?xi1>, tensor<?xbf16>, tensor<?xbf16>) -> tensor<?xbf16>
  return %0 : tensor<?xbf16>
}
