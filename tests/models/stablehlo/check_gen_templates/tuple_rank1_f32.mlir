func.func @main(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> (tensor<?xf32>, tensor<?xf32>) {
  %tup = "stablehlo.tuple"(%arg0, %arg1) : (tensor<?xf32>, tensor<?xf32>) -> tuple<tensor<?xf32>, tensor<?xf32>>
  %0 = "stablehlo.get_tuple_element"(%tup) {index = 0 : i32} : (tuple<tensor<?xf32>, tensor<?xf32>>) -> tensor<?xf32>
  %1 = "stablehlo.get_tuple_element"(%tup) {index = 1 : i32} : (tuple<tensor<?xf32>, tensor<?xf32>>) -> tensor<?xf32>
  return %0, %1 : tensor<?xf32>, tensor<?xf32>
}
