#map = affine_map<(d0) -> (d0)>
func.func @main(%arg0: tensor<?xi32>, %arg1: tensor<?xi32>) -> tensor<?xi32> {
  %0 = linalg.generic {
    indexing_maps = [#map, #map, #map],
    iterator_types = ["parallel"]
  } ins(%arg0, %arg1 : tensor<?xi32>, tensor<?xi32>) outs(%arg0 : tensor<?xi32>) {
  ^bb0(%in: i32, %in_0: i32, %out: i32):
    %1 = arith.addi %in, %in_0 : i32
    linalg.yield %1 : i32
  } -> tensor<?xi32>
  return %0 : tensor<?xi32>
}
