func.func @main(%arg0: tensor<?x?x?xbf16>) -> tensor<?x?x?xbf16> {
  %0 = "stablehlo.transpose"(%arg0) {permutation = array<i64: 1, 2, 0>} : (tensor<?x?x?xbf16>) -> tensor<?x?x?xbf16>
  return %0 : tensor<?x?x?xbf16>
}
