func.func @main(%val: f32, %dim0: index, %dim1: index) -> tensor<?x?xf32> {
  %alloc = tensor.empty(%dim0, %dim1) : tensor<?x?xf32>
  %res = linalg.fill ins(%val : f32) outs(%alloc : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %res : tensor<?x?xf32>
}
