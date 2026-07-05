#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
func.func @main(%arg0: tensor<?x?xf32>) -> tensor<?xf32> {
  %c0 = arith.constant 0.0 : f32
  %c0_idx = arith.constant 0 : index
  %m = tensor.dim %arg0, %c0_idx : tensor<?x?xf32>
  %empty = tensor.empty(%m) : tensor<?xf32>
  %fill = linalg.fill ins(%c0 : f32) outs(%empty : tensor<?xf32>) -> tensor<?xf32>
  %0 = linalg.generic {
    indexing_maps = [#map0, #map1],
    iterator_types = ["parallel", "reduction"]
  } ins(%arg0 : tensor<?x?xf32>) outs(%fill : tensor<?xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1 = arith.addf %in, %out : f32
    linalg.yield %1 : f32
  } -> tensor<?xf32>
  return %0 : tensor<?xf32>
}
