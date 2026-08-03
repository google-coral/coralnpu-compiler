#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func.func @main(%arg0: tensor<?x?x?xi8>, %arg1: tensor<?x?x?xi8>) -> tensor<?x?x?xi8> {
  %0 = linalg.generic {
    indexing_maps = [#map, #map, #map],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<?x?x?xi8>, tensor<?x?x?xi8>) outs(%arg0 : tensor<?x?x?xi8>) {
  ^bb0(%in: i8, %in_0: i8, %out: i8):
    %1 = arith.addi %in, %in_0 : i8
    linalg.yield %1 : i8
  } -> tensor<?x?x?xi8>
  return %0 : tensor<?x?x?xi8>
}
