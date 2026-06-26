func.func @main(%arg0: tensor<?xf32>) -> tensor<?x?xf32> {
  %dim0_0d_i32 = "stablehlo.get_dimension_size"(%arg0) {dimension = 0 : i64} : (tensor<?xf32>) -> tensor<i32>
  %dim0_0d_i64 = stablehlo.convert %dim0_0d_i32 : (tensor<i32>) -> tensor<i64>
  %dim0_1d = stablehlo.reshape %dim0_0d_i64 : (tensor<i64>) -> tensor<1xi64>
  %c_static_0d = stablehlo.constant dense<4> : tensor<i64>
  %c_static_1d = stablehlo.reshape %c_static_0d : (tensor<i64>) -> tensor<1xi64>
  %shape = stablehlo.concatenate %c_static_1d, %dim0_1d, dim = 0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
  %0 = "stablehlo.dynamic_broadcast_in_dim"(%arg0, %shape) {
    broadcast_dimensions = array<i64: 1>
  } : (tensor<?xf32>, tensor<2xi64>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
