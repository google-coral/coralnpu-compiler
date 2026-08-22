func.func @main(%arg0: tensor<?xbf16>) -> tensor<?xbf16> {
  %0 = stablehlo.negate %arg0 : tensor<?xbf16>
  return %0 : tensor<?xbf16>
}
