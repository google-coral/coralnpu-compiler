func.func @main(%operand: tensor<?x?x?x?xf32>, %scale: tensor<?xf32>, %offset: tensor<?xf32>, %mean: tensor<?xf32>, %variance: tensor<?xf32>) -> tensor<?x?x?x?xf32> {
  %0 = "stablehlo.batch_norm_inference"(%operand, %scale, %offset, %mean, %variance) {
    epsilon = 1.000000e-03 : f32,
    feature_index = 3 : i64
  } : (tensor<?x?x?x?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}
