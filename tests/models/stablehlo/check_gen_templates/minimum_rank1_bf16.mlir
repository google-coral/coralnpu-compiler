func.func @main(%arg0: tensor<?xbf16>, %arg1: tensor<?xbf16>) -> tensor<?xbf16> {
  %0 = stablehlo.minimum %arg0, %arg1 : tensor<?xbf16>
  return %0 : tensor<?xbf16>
}
