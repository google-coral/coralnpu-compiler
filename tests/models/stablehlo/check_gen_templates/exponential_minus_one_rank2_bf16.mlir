func.func @main(%arg0: tensor<?x?xbf16>) -> tensor<?x?xbf16> {
  %0 = stablehlo.exponential_minus_one %arg0 : (tensor<?x?xbf16>) -> tensor<?x?xbf16>
  return %0 : tensor<?x?xbf16>
}
