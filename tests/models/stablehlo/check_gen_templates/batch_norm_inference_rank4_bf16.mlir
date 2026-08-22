func.func @main(%operand: tensor<?x?x?x?xbf16>, %scale: tensor<?xbf16>, %offset: tensor<?xbf16>, %mean: tensor<?xbf16>, %variance: tensor<?xbf16>) -> tensor<?x?x?x?xbf16> {
  %0 = "stablehlo.batch_norm_inference"(%operand, %scale, %offset, %mean, %variance) {
    epsilon = 1.000000e-03 : f32,
    feature_index = 3 : i64
  } : (tensor<?x?x?x?xbf16>, tensor<?xbf16>, tensor<?xbf16>, tensor<?xbf16>, tensor<?xbf16>) -> tensor<?x?x?x?xbf16>
  return %0 : tensor<?x?x?x?xbf16>
}
