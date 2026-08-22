func.func @main(%arg0: tensor<?xbf16>, %arg1: tensor<?xbf16>) -> (tensor<?xbf16>, tensor<?xbf16>) {
  %tup = "stablehlo.tuple"(%arg0, %arg1) : (tensor<?xbf16>, tensor<?xbf16>) -> tuple<tensor<?xbf16>, tensor<?xbf16>>
  %0 = "stablehlo.get_tuple_element"(%tup) {index = 0 : i32} : (tuple<tensor<?xbf16>, tensor<?xbf16>>) -> tensor<?xbf16>
  %1 = "stablehlo.get_tuple_element"(%tup) {index = 1 : i32} : (tuple<tensor<?xbf16>, tensor<?xbf16>>) -> tensor<?xbf16>
  return %0, %1 : tensor<?xbf16>, tensor<?xbf16>
}
