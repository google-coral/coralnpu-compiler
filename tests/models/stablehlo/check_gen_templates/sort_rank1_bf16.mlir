func.func @main(%input: tensor<?xbf16>) -> tensor<?xbf16> {
  %0 = "stablehlo.sort"(%input) ({
  ^bb0(%lhs: tensor<bf16>, %rhs: tensor<bf16>):
     %cond = "stablehlo.compare"(%lhs, %rhs) {comparison_direction = #stablehlo<comparison_direction LT>} : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
     "stablehlo.return"(%cond) : (tensor<i1>) -> ()
  }) {
     dimension = 0 : i64,
     is_stable = false
  } : (tensor<?xbf16>) -> tensor<?xbf16>
  return %0 : tensor<?xbf16>
}
