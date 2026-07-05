#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func.func @main(%arg0: tensor<?x?x?xi32>, %arg1: tensor<?x?x?xi32>) -> tensor<?x?x?xi32> {
  %0 = linalg.generic {
    indexing_maps = [#map, #map, #map],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<?x?x?xi32>, tensor<?x?x?xi32>) outs(%arg0 : tensor<?x?x?xi32>) {
  ^bb0(%in: i32, %in_0: i32, %out: i32):
    %1 = arith.addi %in, %in_0 : i32
    linalg.yield %1 : i32
  } -> tensor<?x?x?xi32>
  return %0 : tensor<?x?x?xi32>
}
