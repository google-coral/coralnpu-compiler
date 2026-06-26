func.func @main(%arg0: tensor<?xi8>) -> tensor<4xi8> {
  %0 = "stablehlo.slice"(%arg0) {
    start_indices = array<i64: 2>,
    limit_indices = array<i64: 6>,
    strides = array<i64: 1>
  } : (tensor<?xi8>) -> tensor<4xi8>
  return %0 : tensor<4xi8>
}
