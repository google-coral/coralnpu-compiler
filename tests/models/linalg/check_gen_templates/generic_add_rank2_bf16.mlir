#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @main(%arg0: tensor<?x?xbf16>, %arg1: tensor<?x?xbf16>) -> tensor<?x?xbf16> {
  %0 = linalg.generic {
    indexing_maps = [#map, #map, #map],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<?x?xbf16>, tensor<?x?xbf16>) outs(%arg0 : tensor<?x?xbf16>) {
  ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
    %1 = arith.addf %in, %in_0 : bf16
    linalg.yield %1 : bf16
  } -> tensor<?x?xbf16>
  return %0 : tensor<?x?xbf16>
}
