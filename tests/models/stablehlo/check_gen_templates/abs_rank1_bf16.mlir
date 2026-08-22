func.func @main(%arg0: tensor<?xbf16>) -> tensor<?xbf16> {
  %0 = stablehlo.abs %arg0 : tensor<?xbf16>
  return %0 : tensor<?xbf16>
}
