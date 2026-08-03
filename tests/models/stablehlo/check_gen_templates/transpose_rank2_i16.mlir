func.func @main(%arg0: tensor<?x?xi16>) -> tensor<?x?xi16> {
  %0 = "stablehlo.transpose"(%arg0) {permutation = array<i64: 1, 0>} : (tensor<?x?xi16>) -> tensor<?x?xi16>
  return %0 : tensor<?x?xi16>
}
