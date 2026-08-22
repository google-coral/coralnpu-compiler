func.func @main(%arg0: tensor<?x?x?xbf16>, %arg1: tensor<?x?x?xbf16>) -> tensor<?x?x?xbf16> {
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<?x?x?xbf16>, tensor<?x?x?xbf16>) -> tensor<?x?x?xbf16>
  return %0 : tensor<?x?x?xbf16>
}
