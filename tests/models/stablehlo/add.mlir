func.func @main(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>) -> tensor<8xi32> {
  %add = stablehlo.add %arg0, %arg1 : tensor<8xi32>
  return %add : tensor<8xi32>
}
