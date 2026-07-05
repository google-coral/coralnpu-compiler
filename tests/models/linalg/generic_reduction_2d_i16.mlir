#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
func.func @main(%arg0: tensor<?x?xi16>) -> tensor<?xi16> {
  %c0 = arith.constant 0 : i16
  %c0_idx = arith.constant 0 : index
  %m = tensor.dim %arg0, %c0_idx : tensor<?x?xi16>
  %empty = tensor.empty(%m) : tensor<?xi16>
  %fill = linalg.fill ins(%c0 : i16) outs(%empty : tensor<?xi16>) -> tensor<?xi16>
  %0 = linalg.generic {
    indexing_maps = [#map0, #map1],
    iterator_types = ["parallel", "reduction"]
  } ins(%arg0 : tensor<?x?xi16>) outs(%fill : tensor<?xi16>) {
  ^bb0(%in: i16, %out: i16):
    %1 = arith.addi %in, %out : i16
    linalg.yield %1 : i16
  } -> tensor<?xi16>
  return %0 : tensor<?xi16>
}
