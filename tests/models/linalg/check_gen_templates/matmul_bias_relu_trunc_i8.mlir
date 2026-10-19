// An i8 x i8 -> i32 matmul with a fused bias add, relu and truncation to i8.
// On CoralNPU, it covers the DTCM promotion of the bias tile, the output
// staging of a fused consumer, and (with a large K) the split-K accumulator.
#map = affine_map<(d0, d1) -> (d0, d1)>
#bias_map = affine_map<(d0, d1) -> (d1)>
func.func @main(%arg0: tensor<?x?xi8>, %arg1: tensor<?x?xi8>, %arg2: tensor<?xi32>) -> tensor<?x?xi8> {
  %c0 = arith.constant 0 : i32
  %c0_idx = arith.constant 0 : index
  %c1_idx = arith.constant 1 : index
  %m = tensor.dim %arg0, %c0_idx : tensor<?x?xi8>
  %n = tensor.dim %arg1, %c1_idx : tensor<?x?xi8>
  %empty = tensor.empty(%m, %n) : tensor<?x?xi32>
  %fill = linalg.fill ins(%c0 : i32) outs(%empty : tensor<?x?xi32>) -> tensor<?x?xi32>
  %matmul = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xi8>, tensor<?x?xi8>) outs(%fill : tensor<?x?xi32>) -> tensor<?x?xi32>
  %out = tensor.empty(%m, %n) : tensor<?x?xi8>
  %0 = linalg.generic {
    indexing_maps = [#map, #bias_map, #map],
    iterator_types = ["parallel", "parallel"]
  } ins(%matmul, %arg2 : tensor<?x?xi32>, tensor<?xi32>) outs(%out : tensor<?x?xi8>) {
  ^bb0(%acc: i32, %bias: i32, %o: i8):
    %sum = arith.addi %acc, %bias : i32
    %relu = arith.maxsi %sum, %c0 : i32
    %trunc = arith.trunci %relu : i32 to i8
    linalg.yield %trunc : i8
  } -> tensor<?x?xi8>
  return %0 : tensor<?x?xi8>
}
