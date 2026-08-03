func.func @main(%arg0: tensor<?x?xi32>) -> tensor<4x8xi32> {
  %0 = stablehlo.iota dim = 1 : tensor<4x8xi32>
  return %0 : tensor<4x8xi32>
}
