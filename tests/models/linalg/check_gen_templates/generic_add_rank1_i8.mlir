#map = affine_map<(d0) -> (d0)>
func.func @main(%arg0: tensor<?xi8>, %arg1: tensor<?xi8>) -> tensor<?xi8> {
  %0 = linalg.generic {
    indexing_maps = [#map, #map, #map],
    iterator_types = ["parallel"]
  } ins(%arg0, %arg1 : tensor<?xi8>, tensor<?xi8>) outs(%arg0 : tensor<?xi8>) {
  ^bb0(%in: i8, %in_0: i8, %out: i8):
    %1 = arith.addi %in, %in_0 : i8
    linalg.yield %1 : i8
  } -> tensor<?xi8>
  return %0 : tensor<?xi8>
}
