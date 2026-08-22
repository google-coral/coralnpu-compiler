func.func @main(%arg0: tensor<?x?x?xbf16>) -> tensor<?x?x?xbf16> {
  %0 = stablehlo.exponential %arg0 : (tensor<?x?x?xbf16>) -> tensor<?x?x?xbf16>
  return %0 : tensor<?x?x?xbf16>
}
