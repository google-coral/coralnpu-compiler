func.func @main(%arg0: tensor<?x?x?xbf16>) -> tensor<?x?x?xbf16> {
  %0 = stablehlo.sign %arg0 : (tensor<?x?x?xbf16>) -> tensor<?x?x?xbf16>
  return %0 : tensor<?x?x?xbf16>
}
