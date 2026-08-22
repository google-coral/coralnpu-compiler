func.func @main(%arg0: tensor<?x?xbf16>) -> tensor<?x?xbf16> {
  %0 = stablehlo.cbrt %arg0 : (tensor<?x?xbf16>) -> tensor<?x?xbf16>
  return %0 : tensor<?x?xbf16>
}
