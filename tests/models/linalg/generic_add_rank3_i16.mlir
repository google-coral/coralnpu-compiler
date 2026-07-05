#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func.func @main(%arg0: tensor<?x?x?xi16>, %arg1: tensor<?x?x?xi16>) -> tensor<?x?x?xi16> {
  %0 = linalg.generic {
    indexing_maps = [#map, #map, #map],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<?x?x?xi16>, tensor<?x?x?xi16>) outs(%arg0 : tensor<?x?x?xi16>) {
  ^bb0(%in: i16, %in_0: i16, %out: i16):
    %1 = arith.addi %in, %in_0 : i16
    linalg.yield %1 : i16
  } -> tensor<?x?x?xi16>
  return %0 : tensor<?x?x?xi16>
}
