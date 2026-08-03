func.func @main(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = "stablehlo.transpose"(%arg0) {permutation = array<i64: 1, 0>} : (tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
