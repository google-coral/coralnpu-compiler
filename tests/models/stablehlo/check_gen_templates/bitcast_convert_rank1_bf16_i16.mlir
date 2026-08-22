func.func @main(%arg0: tensor<?xbf16>) -> tensor<?xi16> {
  %0 = stablehlo.bitcast_convert %arg0 : (tensor<?xbf16>) -> tensor<?xi16>
  return %0 : tensor<?xi16>
}
