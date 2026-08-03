func.func @main(%arg0: tensor<?x?x?xi16>) -> tensor<1x2x3xi16> {
  %0 = "stablehlo.slice"(%arg0) {
    start_indices = array<i64: 1, 1, 1>,
    limit_indices = array<i64: 2, 3, 4>,
    strides = array<i64: 1, 1, 1>
  } : (tensor<?x?x?xi16>) -> tensor<1x2x3xi16>
  return %0 : tensor<1x2x3xi16>
}
