// RUN: %template_path
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @main(%arg0: tensor<?x?xi32>, %arg1: tensor<?x?xi32>) -> (tensor<?x?xi32>, tensor<?x?xi32>) {
  %c0_idx = arith.constant 0 : index
  %c1_idx = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0_idx : tensor<?x?xi32>
  %d1 = tensor.dim %arg0, %c1_idx : tensor<?x?xi32>

  %empty0 = tensor.empty(%d0, %d1) : tensor<?x?xi32>
  %empty1 = tensor.empty(%d0, %d1) : tensor<?x?xi32>
  
  %0, %1 = linalg.generic {
    indexing_maps = [#map, #map, #map, #map],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<?x?xi32>, tensor<?x?xi32>)
    outs(%empty0, %empty1 : tensor<?x?xi32>, tensor<?x?xi32>) {
  ^bb0(%in0: i32, %in1: i32, %out0: i32, %out1: i32):
    %add = arith.addi %in0, %in1 : i32
    %sub = arith.subi %in0, %in1 : i32
    linalg.yield %add, %sub : i32, i32
  } -> (tensor<?x?xi32>, tensor<?x?xi32>)
  
  return %0, %1 : tensor<?x?xi32>, tensor<?x?xi32>
}
