#map = affine_map<(d0) -> (d0)>
func.func @main(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
  %0 = linalg.generic {
    indexing_maps = [#map, #map, #map],
    iterator_types = ["parallel"]
  } ins(%arg0, %arg1 : tensor<?xf32>, tensor<?xf32>) outs(%arg0 : tensor<?xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.addf %in, %in_0 : f32
    linalg.yield %1 : f32
  } -> tensor<?xf32>
  return %0 : tensor<?xf32>
}
