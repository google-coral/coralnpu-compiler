// RUN: %template_path
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @main(%arg0: tensor<?x?xbf16>, %arg1: tensor<?x?xbf16>) -> (tensor<?x?xbf16>, tensor<?x?xbf16>) {
  %c0_idx = arith.constant 0 : index
  %c1_idx = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0_idx : tensor<?x?xbf16>
  %d1 = tensor.dim %arg0, %c1_idx : tensor<?x?xbf16>

  %empty0 = tensor.empty(%d0, %d1) : tensor<?x?xbf16>
  %empty1 = tensor.empty(%d0, %d1) : tensor<?x?xbf16>
  
  %0, %1 = linalg.generic {
    indexing_maps = [#map, #map, #map, #map],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<?x?xbf16>, tensor<?x?xbf16>)
    outs(%empty0, %empty1 : tensor<?x?xbf16>, tensor<?x?xbf16>) {
  ^bb0(%in0: bf16, %in1: bf16, %out0: bf16, %out1: bf16):
    %add = arith.addf %in0, %in1 : bf16
    %sub = arith.subf %in0, %in1 : bf16
    linalg.yield %add, %sub : bf16, bf16
  } -> (tensor<?x?xbf16>, tensor<?x?xbf16>)
  
  return %0, %1 : tensor<?x?xbf16>, tensor<?x?xbf16>
}
