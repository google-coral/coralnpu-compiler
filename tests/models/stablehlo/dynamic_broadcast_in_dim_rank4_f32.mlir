func.func @main(%arg0: tensor<?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %dim0_0d_i32 = "stablehlo.get_dimension_size"(%arg0) {dimension = 0 : i64} : (tensor<?x?x?xf32>) -> tensor<i32>
  %dim0_0d_i64 = stablehlo.convert %dim0_0d_i32 : (tensor<i32>) -> tensor<i64>
  %dim0_1d = stablehlo.reshape %dim0_0d_i64 : (tensor<i64>) -> tensor<1xi64>
  %dim1_0d_i32 = "stablehlo.get_dimension_size"(%arg0) {dimension = 1 : i64} : (tensor<?x?x?xf32>) -> tensor<i32>
  %dim1_0d_i64 = stablehlo.convert %dim1_0d_i32 : (tensor<i32>) -> tensor<i64>
  %dim1_1d = stablehlo.reshape %dim1_0d_i64 : (tensor<i64>) -> tensor<1xi64>
  %dim2_0d_i32 = "stablehlo.get_dimension_size"(%arg0) {dimension = 2 : i64} : (tensor<?x?x?xf32>) -> tensor<i32>
  %dim2_0d_i64 = stablehlo.convert %dim2_0d_i32 : (tensor<i32>) -> tensor<i64>
  %dim2_1d = stablehlo.reshape %dim2_0d_i64 : (tensor<i64>) -> tensor<1xi64>
  %c_static_0d = stablehlo.constant dense<6> : tensor<i64>
  %c_static_1d = stablehlo.reshape %c_static_0d : (tensor<i64>) -> tensor<1xi64>
  %shape = stablehlo.concatenate %dim0_1d, %dim1_1d, %c_static_1d, %dim2_1d, dim = 0 : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xi64>
  %0 = "stablehlo.dynamic_broadcast_in_dim"(%arg0, %shape) {
    broadcast_dimensions = array<i64: 0, 1, 3>
  } : (tensor<?x?x?xf32>, tensor<4xi64>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}
