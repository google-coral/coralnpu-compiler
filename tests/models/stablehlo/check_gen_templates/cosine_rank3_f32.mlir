func.func @main(%arg0: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> attributes {check.atol = 5.0e-3 : f32} {
  %0 = stablehlo.cosine %arg0 : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
}
