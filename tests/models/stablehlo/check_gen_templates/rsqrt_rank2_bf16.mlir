func.func @main(%arg0: tensor<?x?xbf16>) -> tensor<?x?xbf16> {
  %0 = stablehlo.rsqrt %arg0 : (tensor<?x?xbf16>) -> tensor<?x?xbf16>
  return %0 : tensor<?x?xbf16>
}
