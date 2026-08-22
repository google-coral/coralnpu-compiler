func.func @main(%arg0: tensor<?xbf16>) -> tensor<4x?xbf16> {
  %0 = stablehlo.broadcast %arg0, sizes = [4] : (tensor<?xbf16>) -> tensor<4x?xbf16>
  return %0 : tensor<4x?xbf16>
}
