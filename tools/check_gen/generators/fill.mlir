module {
  func.func @fill_rank_1_i32(%val: i32, %dim0: index) -> tensor<?xi32> {
    %alloc = tensor.empty(%dim0) : tensor<?xi32>
    %res = linalg.fill ins(%val : i32) outs(%alloc : tensor<?xi32>) -> tensor<?xi32>
    return %res : tensor<?xi32>
  }
  func.func @fill_rank_2_i32(%val: i32, %dim0: index, %dim1: index) -> tensor<?x?xi32> {
    %alloc = tensor.empty(%dim0, %dim1) : tensor<?x?xi32>
    %res = linalg.fill ins(%val : i32) outs(%alloc : tensor<?x?xi32>) -> tensor<?x?xi32>
    return %res : tensor<?x?xi32>
  }
  func.func @fill_rank_3_i32(%val: i32, %dim0: index, %dim1: index, %dim2: index) -> tensor<?x?x?xi32> {
    %alloc = tensor.empty(%dim0, %dim1, %dim2) : tensor<?x?x?xi32>
    %res = linalg.fill ins(%val : i32) outs(%alloc : tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
    return %res : tensor<?x?x?xi32>
  }
  func.func @fill_rank_4_i32(%val: i32, %dim0: index, %dim1: index, %dim2: index, %dim3: index) -> tensor<?x?x?x?xi32> {
    %alloc = tensor.empty(%dim0, %dim1, %dim2, %dim3) : tensor<?x?x?x?xi32>
    %res = linalg.fill ins(%val : i32) outs(%alloc : tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
    return %res : tensor<?x?x?x?xi32>
  }
  func.func @fill_rank_5_i32(%val: i32, %dim0: index, %dim1: index, %dim2: index, %dim3: index, %dim4: index) -> tensor<?x?x?x?x?xi32> {
    %alloc = tensor.empty(%dim0, %dim1, %dim2, %dim3, %dim4) : tensor<?x?x?x?x?xi32>
    %res = linalg.fill ins(%val : i32) outs(%alloc : tensor<?x?x?x?x?xi32>) -> tensor<?x?x?x?x?xi32>
    return %res : tensor<?x?x?x?x?xi32>
  }
  func.func @fill_rank_6_i32(%val: i32, %dim0: index, %dim1: index, %dim2: index, %dim3: index, %dim4: index, %dim5: index) -> tensor<?x?x?x?x?x?xi32> {
    %alloc = tensor.empty(%dim0, %dim1, %dim2, %dim3, %dim4, %dim5) : tensor<?x?x?x?x?x?xi32>
    %res = linalg.fill ins(%val : i32) outs(%alloc : tensor<?x?x?x?x?x?xi32>) -> tensor<?x?x?x?x?x?xi32>
    return %res : tensor<?x?x?x?x?x?xi32>
  }
  func.func @fill_rank_7_i32(%val: i32, %dim0: index, %dim1: index, %dim2: index, %dim3: index, %dim4: index, %dim5: index, %dim6: index) -> tensor<?x?x?x?x?x?x?xi32> {
    %alloc = tensor.empty(%dim0, %dim1, %dim2, %dim3, %dim4, %dim5, %dim6) : tensor<?x?x?x?x?x?x?xi32>
    %res = linalg.fill ins(%val : i32) outs(%alloc : tensor<?x?x?x?x?x?x?xi32>) -> tensor<?x?x?x?x?x?x?xi32>
    return %res : tensor<?x?x?x?x?x?x?xi32>
  }
  func.func @fill_rank_8_i32(%val: i32, %dim0: index, %dim1: index, %dim2: index, %dim3: index, %dim4: index, %dim5: index, %dim6: index, %dim7: index) -> tensor<?x?x?x?x?x?x?x?xi32> {
    %alloc = tensor.empty(%dim0, %dim1, %dim2, %dim3, %dim4, %dim5, %dim6, %dim7) : tensor<?x?x?x?x?x?x?x?xi32>
    %res = linalg.fill ins(%val : i32) outs(%alloc : tensor<?x?x?x?x?x?x?x?xi32>) -> tensor<?x?x?x?x?x?x?x?xi32>
    return %res : tensor<?x?x?x?x?x?x?x?xi32>
  }
  func.func @fill_rank_9_i32(%val: i32, %dim0: index, %dim1: index, %dim2: index, %dim3: index, %dim4: index, %dim5: index, %dim6: index, %dim7: index, %dim8: index) -> tensor<?x?x?x?x?x?x?x?x?xi32> {
    %alloc = tensor.empty(%dim0, %dim1, %dim2, %dim3, %dim4, %dim5, %dim6, %dim7, %dim8) : tensor<?x?x?x?x?x?x?x?x?xi32>
    %res = linalg.fill ins(%val : i32) outs(%alloc : tensor<?x?x?x?x?x?x?x?x?xi32>) -> tensor<?x?x?x?x?x?x?x?x?xi32>
    return %res : tensor<?x?x?x?x?x?x?x?x?xi32>
  }
  func.func @fill_rank_10_i32(%val: i32, %dim0: index, %dim1: index, %dim2: index, %dim3: index, %dim4: index, %dim5: index, %dim6: index, %dim7: index, %dim8: index, %dim9: index) -> tensor<?x?x?x?x?x?x?x?x?x?xi32> {
    %alloc = tensor.empty(%dim0, %dim1, %dim2, %dim3, %dim4, %dim5, %dim6, %dim7, %dim8, %dim9) : tensor<?x?x?x?x?x?x?x?x?x?xi32>
    %res = linalg.fill ins(%val : i32) outs(%alloc : tensor<?x?x?x?x?x?x?x?x?x?xi32>) -> tensor<?x?x?x?x?x?x?x?x?x?xi32>
    return %res : tensor<?x?x?x?x?x?x?x?x?x?xi32>
  }
}
