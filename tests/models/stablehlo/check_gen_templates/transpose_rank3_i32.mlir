func.func @main(%arg0: tensor<?x?x?xi32>) -> tensor<?x?x?xi32> {
  %0 = "stablehlo.transpose"(%arg0) {permutation = array<i64: 1, 2, 0>} : (tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  return %0 : tensor<?x?x?xi32>
}
