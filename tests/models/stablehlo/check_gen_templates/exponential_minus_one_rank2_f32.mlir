func.func @main(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = stablehlo.exponential_minus_one %arg0 : (tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
