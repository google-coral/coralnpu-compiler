#map = affine_map<(d0) -> (d0)>
func.func @main(%arg0: tensor<?xi16>, %arg1: tensor<?xi16>) -> tensor<?xi16> {
  %0 = linalg.generic {
    indexing_maps = [#map, #map, #map],
    iterator_types = ["parallel"]
  } ins(%arg0, %arg1 : tensor<?xi16>, tensor<?xi16>) outs(%arg0 : tensor<?xi16>) {
  ^bb0(%in: i16, %in_0: i16, %out: i16):
    %1 = arith.addi %in, %in_0 : i16
    linalg.yield %1 : i16
  } -> tensor<?xi16>
  return %0 : tensor<?xi16>
}
