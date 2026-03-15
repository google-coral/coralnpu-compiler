func.func @add(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>) -> tensor<8xi32> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<8xi32>
  return %0 : tensor<8xi32>
}
