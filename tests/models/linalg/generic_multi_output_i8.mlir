// RUN: %template_path
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @main(%arg0: tensor<?x?xi8>, %arg1: tensor<?x?xi8>) -> (tensor<?x?xi8>, tensor<?x?xi8>) {
  %c0_idx = arith.constant 0 : index
  %c1_idx = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0_idx : tensor<?x?xi8>
  %d1 = tensor.dim %arg0, %c1_idx : tensor<?x?xi8>

  %empty0 = tensor.empty(%d0, %d1) : tensor<?x?xi8>
  %empty1 = tensor.empty(%d0, %d1) : tensor<?x?xi8>
  
  %0, %1 = linalg.generic {
    indexing_maps = [#map, #map, #map, #map],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<?x?xi8>, tensor<?x?xi8>)
    outs(%empty0, %empty1 : tensor<?x?xi8>, tensor<?x?xi8>) {
  ^bb0(%in0: i8, %in1: i8, %out0: i8, %out1: i8):
    %add = arith.addi %in0, %in1 : i8
    %sub = arith.subi %in0, %in1 : i8
    linalg.yield %add, %sub : i8, i8
  } -> (tensor<?x?xi8>, tensor<?x?xi8>)
  
  return %0, %1 : tensor<?x?xi8>, tensor<?x?xi8>
}
