func.func @main(%arg0: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %0 = stablehlo.rsqrt %arg0 : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
}
