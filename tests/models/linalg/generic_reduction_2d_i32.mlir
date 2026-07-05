#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
func.func @main(%arg0: tensor<?x?xi32>) -> tensor<?xi32> {
  %c0 = arith.constant 0 : i32
  %c0_idx = arith.constant 0 : index
  %m = tensor.dim %arg0, %c0_idx : tensor<?x?xi32>
  %empty = tensor.empty(%m) : tensor<?xi32>
  %fill = linalg.fill ins(%c0 : i32) outs(%empty : tensor<?xi32>) -> tensor<?xi32>
  %0 = linalg.generic {
    indexing_maps = [#map0, #map1],
    iterator_types = ["parallel", "reduction"]
  } ins(%arg0 : tensor<?x?xi32>) outs(%fill : tensor<?xi32>) {
  ^bb0(%in: i32, %out: i32):
    %1 = arith.addi %in, %out : i32
    linalg.yield %1 : i32
  } -> tensor<?xi32>
  return %0 : tensor<?xi32>
}
