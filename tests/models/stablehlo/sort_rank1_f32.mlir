func.func @main(%input: tensor<?xf32>) -> tensor<?xf32> {
  %0 = "stablehlo.sort"(%input) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
     %cond = "stablehlo.compare"(%lhs, %rhs) {comparison_direction = #stablehlo<comparison_direction LT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
     "stablehlo.return"(%cond) : (tensor<i1>) -> ()
  }) {
     dimension = 0 : i64,
     is_stable = false
  } : (tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}
