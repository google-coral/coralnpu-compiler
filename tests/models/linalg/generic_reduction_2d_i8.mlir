#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
func.func @main(%arg0: tensor<?x?xi8>) -> tensor<?xi8> {
  %c0 = arith.constant 0 : i8
  %c0_idx = arith.constant 0 : index
  %m = tensor.dim %arg0, %c0_idx : tensor<?x?xi8>
  %empty = tensor.empty(%m) : tensor<?xi8>
  %fill = linalg.fill ins(%c0 : i8) outs(%empty : tensor<?xi8>) -> tensor<?xi8>
  %0 = linalg.generic {
    indexing_maps = [#map0, #map1],
    iterator_types = ["parallel", "reduction"]
  } ins(%arg0 : tensor<?x?xi8>) outs(%fill : tensor<?xi8>) {
  ^bb0(%in: i8, %out: i8):
    %1 = arith.addi %in, %out : i8
    linalg.yield %1 : i8
  } -> tensor<?xi8>
  return %0 : tensor<?xi8>
}
