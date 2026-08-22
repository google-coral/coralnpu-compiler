func.func @main(%arg0: tensor<?x?xbf16>) -> tensor<2x4xbf16> {
  %0 = "stablehlo.slice"(%arg0) {
    start_indices = array<i64: 1, 2>,
    limit_indices = array<i64: 3, 6>,
    strides = array<i64: 1, 1>
  } : (tensor<?x?xbf16>) -> tensor<2x4xbf16>
  return %0 : tensor<2x4xbf16>
}
