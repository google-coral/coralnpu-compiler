#map2 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @positive_sequential_rank_2_i8(%dim0: index, %dim1: index) -> tensor<?x?xi8> {
    %alloc = tensor.empty(%dim0, %dim1) : tensor<?x?xi8>
    %c1 = arith.constant 1 : i32
    %c5 = arith.constant 5 : i32
    %res = linalg.generic {
      indexing_maps = [#map2],
      iterator_types = ["parallel", "parallel"]
    } outs(%alloc : tensor<?x?xi8>) {
    ^bb0(%out: i8):
      %idx0 = linalg.index 0 : index
      %idx0_i32 = arith.index_cast %idx0 : index to i32
      %idx1 = linalg.index 1 : index
      %idx1_i32 = arith.index_cast %idx1 : index to i32
      %dim1_i32 = arith.index_cast %dim1 : index to i32
      %tmp1 = arith.muli %idx0_i32, %dim1_i32 : i32
      %val1 = arith.addi %idx1_i32, %tmp1 : i32
      %val2 = arith.remui %val1, %c5 : i32
      %val3 = arith.addi %val2, %c1 : i32
      %val_i8 = arith.trunci %val3 : i32 to i8
      linalg.yield %val_i8 : i8
    } -> tensor<?x?xi8>
    return %res : tensor<?x?xi8>
  }

  func.func @positive_sequential_rank_2_i16(%dim0: index, %dim1: index) -> tensor<?x?xi16> {
    %alloc = tensor.empty(%dim0, %dim1) : tensor<?x?xi16>
    %c1 = arith.constant 1 : i32
    %c5 = arith.constant 5 : i32
    %res = linalg.generic {
      indexing_maps = [#map2],
      iterator_types = ["parallel", "parallel"]
    } outs(%alloc : tensor<?x?xi16>) {
    ^bb0(%out: i16):
      %idx0 = linalg.index 0 : index
      %idx0_i32 = arith.index_cast %idx0 : index to i32
      %idx1 = linalg.index 1 : index
      %idx1_i32 = arith.index_cast %idx1 : index to i32
      %dim1_i32 = arith.index_cast %dim1 : index to i32
      %tmp1 = arith.muli %idx0_i32, %dim1_i32 : i32
      %val1 = arith.addi %idx1_i32, %tmp1 : i32
      %val2 = arith.remui %val1, %c5 : i32
      %val3 = arith.addi %val2, %c1 : i32
      %val_i16 = arith.trunci %val3 : i32 to i16
      linalg.yield %val_i16 : i16
    } -> tensor<?x?xi16>
    return %res : tensor<?x?xi16>
  }

  func.func @positive_sequential_rank_2_i32(%dim0: index, %dim1: index) -> tensor<?x?xi32> {
    %alloc = tensor.empty(%dim0, %dim1) : tensor<?x?xi32>
    %c1 = arith.constant 1 : i32
    %c5 = arith.constant 5 : i32
    %res = linalg.generic {
      indexing_maps = [#map2],
      iterator_types = ["parallel", "parallel"]
    } outs(%alloc : tensor<?x?xi32>) {
    ^bb0(%out: i32):
      %idx0 = linalg.index 0 : index
      %idx0_i32 = arith.index_cast %idx0 : index to i32
      %idx1 = linalg.index 1 : index
      %idx1_i32 = arith.index_cast %idx1 : index to i32
      %dim1_i32 = arith.index_cast %dim1 : index to i32
      %tmp1 = arith.muli %idx0_i32, %dim1_i32 : i32
      %val1 = arith.addi %idx1_i32, %tmp1 : i32
      %val2 = arith.remui %val1, %c5 : i32
      %val3 = arith.addi %val2, %c1 : i32
      linalg.yield %val3 : i32
    } -> tensor<?x?xi32>
    return %res : tensor<?x?xi32>
  }

  func.func @positive_sequential_rank_2_f32(%dim0: index, %dim1: index) -> tensor<?x?xf32> {
    %alloc = tensor.empty(%dim0, %dim1) : tensor<?x?xf32>
    %c1 = arith.constant 1 : i32
    %c5 = arith.constant 5 : i32
    %res = linalg.generic {
      indexing_maps = [#map2],
      iterator_types = ["parallel", "parallel"]
    } outs(%alloc : tensor<?x?xf32>) {
    ^bb0(%out: f32):
      %idx0 = linalg.index 0 : index
      %idx0_i32 = arith.index_cast %idx0 : index to i32
      %idx1 = linalg.index 1 : index
      %idx1_i32 = arith.index_cast %idx1 : index to i32
      %dim1_i32 = arith.index_cast %dim1 : index to i32
      %tmp1 = arith.muli %idx0_i32, %dim1_i32 : i32
      %val1 = arith.addi %idx1_i32, %tmp1 : i32
      %val2 = arith.remui %val1, %c5 : i32
      %val3 = arith.addi %val2, %c1 : i32
      %val_f32 = arith.sitofp %val3 : i32 to f32
      linalg.yield %val_f32 : f32
    } -> tensor<?x?xf32>
    return %res : tensor<?x?xf32>
  }
}
