// An f32 matmul whose operands are dequantized from i8 (sitofp and scale). IREE
// fuses the dequantizations into the matmul dispatch. On CoralNPU, it covers the
// DTCM promotion of narrow tiles: the i8 tiles are copied before the fused
// producers. The scales keep all sums exact in f32.
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @main(%arg0: tensor<?x?xi8>, %arg1: tensor<?x?xi8>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0.0 : f32
  %lhs_scale = arith.constant 0.5 : f32
  %rhs_scale = arith.constant 0.25 : f32
  %c0_idx = arith.constant 0 : index
  %c1_idx = arith.constant 1 : index
  %m = tensor.dim %arg0, %c0_idx : tensor<?x?xi8>
  %k = tensor.dim %arg0, %c1_idx : tensor<?x?xi8>
  %n = tensor.dim %arg1, %c1_idx : tensor<?x?xi8>
  %lhs_empty = tensor.empty(%m, %k) : tensor<?x?xf32>
  %lhs = linalg.generic {
    indexing_maps = [#map, #map],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0 : tensor<?x?xi8>) outs(%lhs_empty : tensor<?x?xf32>) {
  ^bb0(%in: i8, %o: f32):
    %f = arith.sitofp %in : i8 to f32
    %scaled = arith.mulf %f, %lhs_scale : f32
    linalg.yield %scaled : f32
  } -> tensor<?x?xf32>
  %rhs_empty = tensor.empty(%k, %n) : tensor<?x?xf32>
  %rhs = linalg.generic {
    indexing_maps = [#map, #map],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg1 : tensor<?x?xi8>) outs(%rhs_empty : tensor<?x?xf32>) {
  ^bb0(%in: i8, %o: f32):
    %f = arith.sitofp %in : i8 to f32
    %scaled = arith.mulf %f, %rhs_scale : f32
    linalg.yield %scaled : f32
  } -> tensor<?x?xf32>
  %empty = tensor.empty(%m, %n) : tensor<?x?xf32>
  %fill = linalg.fill ins(%c0 : f32) outs(%empty : tensor<?x?xf32>) -> tensor<?x?xf32>
  %0 = linalg.matmul ins(%lhs, %rhs : tensor<?x?xf32>, tensor<?x?xf32>) outs(%fill : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
