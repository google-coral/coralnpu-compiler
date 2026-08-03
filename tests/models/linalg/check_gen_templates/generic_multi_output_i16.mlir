// RUN: %template_path
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @main(%arg0: tensor<?x?xi16>, %arg1: tensor<?x?xi16>) -> (tensor<?x?xi16>, tensor<?x?xi16>) {
  %c0_idx = arith.constant 0 : index
  %c1_idx = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0_idx : tensor<?x?xi16>
  %d1 = tensor.dim %arg0, %c1_idx : tensor<?x?xi16>

  %empty0 = tensor.empty(%d0, %d1) : tensor<?x?xi16>
  %empty1 = tensor.empty(%d0, %d1) : tensor<?x?xi16>
  
  %0, %1 = linalg.generic {
    indexing_maps = [#map, #map, #map, #map],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<?x?xi16>, tensor<?x?xi16>)
    outs(%empty0, %empty1 : tensor<?x?xi16>, tensor<?x?xi16>) {
  ^bb0(%in0: i16, %in1: i16, %out0: i16, %out1: i16):
    %add = arith.addi %in0, %in1 : i16
    %sub = arith.subi %in0, %in1 : i16
    linalg.yield %add, %sub : i16, i16
  } -> (tensor<?x?xi16>, tensor<?x?xi16>)
  
  return %0, %1 : tensor<?x?xi16>, tensor<?x?xi16>
}
