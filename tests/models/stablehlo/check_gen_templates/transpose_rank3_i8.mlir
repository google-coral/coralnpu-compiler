func.func @main(%arg0: tensor<?x?x?xi8>) -> tensor<?x?x?xi8> {
  %0 = "stablehlo.transpose"(%arg0) {permutation = array<i64: 1, 2, 0>} : (tensor<?x?x?xi8>) -> tensor<?x?x?xi8>
  return %0 : tensor<?x?x?xi8>
}
