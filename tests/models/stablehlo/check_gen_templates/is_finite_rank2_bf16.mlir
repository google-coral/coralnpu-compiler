func.func @main(%arg0: tensor<?x?xbf16>) -> tensor<?x?xi1> {
  %0 = stablehlo.is_finite %arg0 : (tensor<?x?xbf16>) -> tensor<?x?xi1>
  return %0 : tensor<?x?xi1>
}
