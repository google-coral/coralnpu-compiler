#map = affine_map<(d0) -> (d0)>
func.func @main(%arg0: tensor<?xbf16>, %arg1: tensor<?xbf16>) -> tensor<?xbf16> {
  %0 = linalg.generic {
    indexing_maps = [#map, #map, #map],
    iterator_types = ["parallel"]
  } ins(%arg0, %arg1 : tensor<?xbf16>, tensor<?xbf16>) outs(%arg0 : tensor<?xbf16>) {
  ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
    %1 = arith.addf %in, %in_0 : bf16
    linalg.yield %1 : bf16
  } -> tensor<?xbf16>
  return %0 : tensor<?xbf16>
}
