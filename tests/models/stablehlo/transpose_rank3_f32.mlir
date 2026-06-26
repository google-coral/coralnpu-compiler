func.func @main(%arg0: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %0 = "stablehlo.transpose"(%arg0) {permutation = array<i64: 1, 2, 0>} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
}
