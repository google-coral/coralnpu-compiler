// RUN: %template_path
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @main(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
  %c0_idx = arith.constant 0 : index
  %c1_idx = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0_idx : tensor<?x?xf32>
  %d1 = tensor.dim %arg0, %c1_idx : tensor<?x?xf32>

  %empty0 = tensor.empty(%d0, %d1) : tensor<?x?xf32>
  %empty1 = tensor.empty(%d0, %d1) : tensor<?x?xf32>
  
  %0, %1 = linalg.generic {
    indexing_maps = [#map, #map, #map, #map],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%empty0, %empty1 : tensor<?x?xf32>, tensor<?x?xf32>) {
  ^bb0(%in0: f32, %in1: f32, %out0: f32, %out1: f32):
    %add = arith.addf %in0, %in1 : f32
    %sub = arith.subf %in0, %in1 : f32
    linalg.yield %add, %sub : f32, f32
  } -> (tensor<?x?xf32>, tensor<?x?xf32>)
  
  return %0, %1 : tensor<?x?xf32>, tensor<?x?xf32>
}
