#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map5 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map6 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
#map7 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d5, d6)>
#map8 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4, d5, d6, d7)>
#map9 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4, d5, d6, d7, d8)>
#map10 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d0, d1, d2, d3, d4, d5, d6, d7, d8, d9)>
module {
  func.func @sequential_rank_1_i32(%dim0: index) -> tensor<?xi32> {
    %alloc = tensor.empty(%dim0) : tensor<?xi32>
    %res = linalg.generic {
      indexing_maps = [#map1],
      iterator_types = ["parallel"]
    } outs(%alloc : tensor<?xi32>) {
    ^bb0(%out: i32):
      %idx0 = linalg.index 0 : index
      %idx0_i32 = arith.index_cast %idx0 : index to i32
      linalg.yield %idx0_i32 : i32
    } -> tensor<?xi32>
    return %res : tensor<?xi32>
  }
  func.func @sequential_rank_1_i1(%dim0: index) -> tensor<?xi1> {
    %alloc = tensor.empty(%dim0) : tensor<?xi1>
    %res = linalg.generic {
      indexing_maps = [#map1],
      iterator_types = ["parallel"]
    } outs(%alloc : tensor<?xi1>) {
    ^bb0(%out: i1):
      %idx0 = linalg.index 0 : index
      %idx0_i32 = arith.index_cast %idx0 : index to i32
      %val_i1 = arith.trunci %idx0_i32 : i32 to i1
      linalg.yield %val_i1 : i1
    } -> tensor<?xi1>
    return %res : tensor<?xi1>
  }
  func.func @sequential_rank_2_i32(%dim0: index, %dim1: index) -> tensor<?x?xi32> {
    %alloc = tensor.empty(%dim0, %dim1) : tensor<?x?xi32>
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
      linalg.yield %val1 : i32
    } -> tensor<?x?xi32>
    return %res : tensor<?x?xi32>
  }
  func.func @sequential_rank_2_i1(%dim0: index, %dim1: index) -> tensor<?x?xi1> {
    %alloc = tensor.empty(%dim0, %dim1) : tensor<?x?xi1>
    %res = linalg.generic {
      indexing_maps = [#map2],
      iterator_types = ["parallel", "parallel"]
    } outs(%alloc : tensor<?x?xi1>) {
    ^bb0(%out: i1):
      %idx0 = linalg.index 0 : index
      %idx0_i32 = arith.index_cast %idx0 : index to i32
      %idx1 = linalg.index 1 : index
      %idx1_i32 = arith.index_cast %idx1 : index to i32
      %dim1_i32 = arith.index_cast %dim1 : index to i32
      %tmp1 = arith.muli %idx0_i32, %dim1_i32 : i32
      %val1 = arith.addi %idx1_i32, %tmp1 : i32
      %val_i1 = arith.trunci %val1 : i32 to i1
      linalg.yield %val_i1 : i1
    } -> tensor<?x?xi1>
    return %res : tensor<?x?xi1>
  }
  func.func @sequential_rank_3_i32(%dim0: index, %dim1: index, %dim2: index) -> tensor<?x?x?xi32> {
    %alloc = tensor.empty(%dim0, %dim1, %dim2) : tensor<?x?x?xi32>
    %res = linalg.generic {
      indexing_maps = [#map3],
      iterator_types = ["parallel", "parallel", "parallel"]
    } outs(%alloc : tensor<?x?x?xi32>) {
    ^bb0(%out: i32):
      %idx0 = linalg.index 0 : index
      %idx0_i32 = arith.index_cast %idx0 : index to i32
      %idx1 = linalg.index 1 : index
      %idx1_i32 = arith.index_cast %idx1 : index to i32
      %idx2 = linalg.index 2 : index
      %idx2_i32 = arith.index_cast %idx2 : index to i32
      %dim1_i32 = arith.index_cast %dim1 : index to i32
      %tmp1 = arith.muli %idx0_i32, %dim1_i32 : i32
      %val1 = arith.addi %idx1_i32, %tmp1 : i32
      %dim2_i32 = arith.index_cast %dim2 : index to i32
      %tmp2 = arith.muli %val1, %dim2_i32 : i32
      %val2 = arith.addi %idx2_i32, %tmp2 : i32
      linalg.yield %val2 : i32
    } -> tensor<?x?x?xi32>
    return %res : tensor<?x?x?xi32>
  }
  func.func @sequential_rank_3_i1(%dim0: index, %dim1: index, %dim2: index) -> tensor<?x?x?xi1> {
    %alloc = tensor.empty(%dim0, %dim1, %dim2) : tensor<?x?x?xi1>
    %res = linalg.generic {
      indexing_maps = [#map3],
      iterator_types = ["parallel", "parallel", "parallel"]
    } outs(%alloc : tensor<?x?x?xi1>) {
    ^bb0(%out: i1):
      %idx0 = linalg.index 0 : index
      %idx0_i32 = arith.index_cast %idx0 : index to i32
      %idx1 = linalg.index 1 : index
      %idx1_i32 = arith.index_cast %idx1 : index to i32
      %idx2 = linalg.index 2 : index
      %idx2_i32 = arith.index_cast %idx2 : index to i32
      %dim1_i32 = arith.index_cast %dim1 : index to i32
      %tmp1 = arith.muli %idx0_i32, %dim1_i32 : i32
      %val1 = arith.addi %idx1_i32, %tmp1 : i32
      %dim2_i32 = arith.index_cast %dim2 : index to i32
      %tmp2 = arith.muli %val1, %dim2_i32 : i32
      %val2 = arith.addi %idx2_i32, %tmp2 : i32
      %val_i1 = arith.trunci %val2 : i32 to i1
      linalg.yield %val_i1 : i1
    } -> tensor<?x?x?xi1>
    return %res : tensor<?x?x?xi1>
  }
  func.func @sequential_rank_4_i32(%dim0: index, %dim1: index, %dim2: index, %dim3: index) -> tensor<?x?x?x?xi32> {
    %alloc = tensor.empty(%dim0, %dim1, %dim2, %dim3) : tensor<?x?x?x?xi32>
    %res = linalg.generic {
      indexing_maps = [#map4],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } outs(%alloc : tensor<?x?x?x?xi32>) {
    ^bb0(%out: i32):
      %idx0 = linalg.index 0 : index
      %idx0_i32 = arith.index_cast %idx0 : index to i32
      %idx1 = linalg.index 1 : index
      %idx1_i32 = arith.index_cast %idx1 : index to i32
      %idx2 = linalg.index 2 : index
      %idx2_i32 = arith.index_cast %idx2 : index to i32
      %idx3 = linalg.index 3 : index
      %idx3_i32 = arith.index_cast %idx3 : index to i32
      %dim1_i32 = arith.index_cast %dim1 : index to i32
      %tmp1 = arith.muli %idx0_i32, %dim1_i32 : i32
      %val1 = arith.addi %idx1_i32, %tmp1 : i32
      %dim2_i32 = arith.index_cast %dim2 : index to i32
      %tmp2 = arith.muli %val1, %dim2_i32 : i32
      %val2 = arith.addi %idx2_i32, %tmp2 : i32
      %dim3_i32 = arith.index_cast %dim3 : index to i32
      %tmp3 = arith.muli %val2, %dim3_i32 : i32
      %val3 = arith.addi %idx3_i32, %tmp3 : i32
      linalg.yield %val3 : i32
    } -> tensor<?x?x?x?xi32>
    return %res : tensor<?x?x?x?xi32>
  }
  func.func @sequential_rank_4_i1(%dim0: index, %dim1: index, %dim2: index, %dim3: index) -> tensor<?x?x?x?xi1> {
    %alloc = tensor.empty(%dim0, %dim1, %dim2, %dim3) : tensor<?x?x?x?xi1>
    %res = linalg.generic {
      indexing_maps = [#map4],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } outs(%alloc : tensor<?x?x?x?xi1>) {
    ^bb0(%out: i1):
      %idx0 = linalg.index 0 : index
      %idx0_i32 = arith.index_cast %idx0 : index to i32
      %idx1 = linalg.index 1 : index
      %idx1_i32 = arith.index_cast %idx1 : index to i32
      %idx2 = linalg.index 2 : index
      %idx2_i32 = arith.index_cast %idx2 : index to i32
      %idx3 = linalg.index 3 : index
      %idx3_i32 = arith.index_cast %idx3 : index to i32
      %dim1_i32 = arith.index_cast %dim1 : index to i32
      %tmp1 = arith.muli %idx0_i32, %dim1_i32 : i32
      %val1 = arith.addi %idx1_i32, %tmp1 : i32
      %dim2_i32 = arith.index_cast %dim2 : index to i32
      %tmp2 = arith.muli %val1, %dim2_i32 : i32
      %val2 = arith.addi %idx2_i32, %tmp2 : i32
      %dim3_i32 = arith.index_cast %dim3 : index to i32
      %tmp3 = arith.muli %val2, %dim3_i32 : i32
      %val3 = arith.addi %idx3_i32, %tmp3 : i32
      %val_i1 = arith.trunci %val3 : i32 to i1
      linalg.yield %val_i1 : i1
    } -> tensor<?x?x?x?xi1>
    return %res : tensor<?x?x?x?xi1>
  }
  func.func @sequential_rank_5_i32(%dim0: index, %dim1: index, %dim2: index, %dim3: index, %dim4: index) -> tensor<?x?x?x?x?xi32> {
    %alloc = tensor.empty(%dim0, %dim1, %dim2, %dim3, %dim4) : tensor<?x?x?x?x?xi32>
    %res = linalg.generic {
      indexing_maps = [#map5],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]
    } outs(%alloc : tensor<?x?x?x?x?xi32>) {
    ^bb0(%out: i32):
      %idx0 = linalg.index 0 : index
      %idx0_i32 = arith.index_cast %idx0 : index to i32
      %idx1 = linalg.index 1 : index
      %idx1_i32 = arith.index_cast %idx1 : index to i32
      %idx2 = linalg.index 2 : index
      %idx2_i32 = arith.index_cast %idx2 : index to i32
      %idx3 = linalg.index 3 : index
      %idx3_i32 = arith.index_cast %idx3 : index to i32
      %idx4 = linalg.index 4 : index
      %idx4_i32 = arith.index_cast %idx4 : index to i32
      %dim1_i32 = arith.index_cast %dim1 : index to i32
      %tmp1 = arith.muli %idx0_i32, %dim1_i32 : i32
      %val1 = arith.addi %idx1_i32, %tmp1 : i32
      %dim2_i32 = arith.index_cast %dim2 : index to i32
      %tmp2 = arith.muli %val1, %dim2_i32 : i32
      %val2 = arith.addi %idx2_i32, %tmp2 : i32
      %dim3_i32 = arith.index_cast %dim3 : index to i32
      %tmp3 = arith.muli %val2, %dim3_i32 : i32
      %val3 = arith.addi %idx3_i32, %tmp3 : i32
      %dim4_i32 = arith.index_cast %dim4 : index to i32
      %tmp4 = arith.muli %val3, %dim4_i32 : i32
      %val4 = arith.addi %idx4_i32, %tmp4 : i32
      linalg.yield %val4 : i32
    } -> tensor<?x?x?x?x?xi32>
    return %res : tensor<?x?x?x?x?xi32>
  }
  func.func @sequential_rank_5_i1(%dim0: index, %dim1: index, %dim2: index, %dim3: index, %dim4: index) -> tensor<?x?x?x?x?xi1> {
    %alloc = tensor.empty(%dim0, %dim1, %dim2, %dim3, %dim4) : tensor<?x?x?x?x?xi1>
    %res = linalg.generic {
      indexing_maps = [#map5],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]
    } outs(%alloc : tensor<?x?x?x?x?xi1>) {
    ^bb0(%out: i1):
      %idx0 = linalg.index 0 : index
      %idx0_i32 = arith.index_cast %idx0 : index to i32
      %idx1 = linalg.index 1 : index
      %idx1_i32 = arith.index_cast %idx1 : index to i32
      %idx2 = linalg.index 2 : index
      %idx2_i32 = arith.index_cast %idx2 : index to i32
      %idx3 = linalg.index 3 : index
      %idx3_i32 = arith.index_cast %idx3 : index to i32
      %idx4 = linalg.index 4 : index
      %idx4_i32 = arith.index_cast %idx4 : index to i32
      %dim1_i32 = arith.index_cast %dim1 : index to i32
      %tmp1 = arith.muli %idx0_i32, %dim1_i32 : i32
      %val1 = arith.addi %idx1_i32, %tmp1 : i32
      %dim2_i32 = arith.index_cast %dim2 : index to i32
      %tmp2 = arith.muli %val1, %dim2_i32 : i32
      %val2 = arith.addi %idx2_i32, %tmp2 : i32
      %dim3_i32 = arith.index_cast %dim3 : index to i32
      %tmp3 = arith.muli %val2, %dim3_i32 : i32
      %val3 = arith.addi %idx3_i32, %tmp3 : i32
      %dim4_i32 = arith.index_cast %dim4 : index to i32
      %tmp4 = arith.muli %val3, %dim4_i32 : i32
      %val4 = arith.addi %idx4_i32, %tmp4 : i32
      %val_i1 = arith.trunci %val4 : i32 to i1
      linalg.yield %val_i1 : i1
    } -> tensor<?x?x?x?x?xi1>
    return %res : tensor<?x?x?x?x?xi1>
  }
  func.func @sequential_rank_6_i32(%dim0: index, %dim1: index, %dim2: index, %dim3: index, %dim4: index, %dim5: index) -> tensor<?x?x?x?x?x?xi32> {
    %alloc = tensor.empty(%dim0, %dim1, %dim2, %dim3, %dim4, %dim5) : tensor<?x?x?x?x?x?xi32>
    %res = linalg.generic {
      indexing_maps = [#map6],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]
    } outs(%alloc : tensor<?x?x?x?x?x?xi32>) {
    ^bb0(%out: i32):
      %idx0 = linalg.index 0 : index
      %idx0_i32 = arith.index_cast %idx0 : index to i32
      %idx1 = linalg.index 1 : index
      %idx1_i32 = arith.index_cast %idx1 : index to i32
      %idx2 = linalg.index 2 : index
      %idx2_i32 = arith.index_cast %idx2 : index to i32
      %idx3 = linalg.index 3 : index
      %idx3_i32 = arith.index_cast %idx3 : index to i32
      %idx4 = linalg.index 4 : index
      %idx4_i32 = arith.index_cast %idx4 : index to i32
      %idx5 = linalg.index 5 : index
      %idx5_i32 = arith.index_cast %idx5 : index to i32
      %dim1_i32 = arith.index_cast %dim1 : index to i32
      %tmp1 = arith.muli %idx0_i32, %dim1_i32 : i32
      %val1 = arith.addi %idx1_i32, %tmp1 : i32
      %dim2_i32 = arith.index_cast %dim2 : index to i32
      %tmp2 = arith.muli %val1, %dim2_i32 : i32
      %val2 = arith.addi %idx2_i32, %tmp2 : i32
      %dim3_i32 = arith.index_cast %dim3 : index to i32
      %tmp3 = arith.muli %val2, %dim3_i32 : i32
      %val3 = arith.addi %idx3_i32, %tmp3 : i32
      %dim4_i32 = arith.index_cast %dim4 : index to i32
      %tmp4 = arith.muli %val3, %dim4_i32 : i32
      %val4 = arith.addi %idx4_i32, %tmp4 : i32
      %dim5_i32 = arith.index_cast %dim5 : index to i32
      %tmp5 = arith.muli %val4, %dim5_i32 : i32
      %val5 = arith.addi %idx5_i32, %tmp5 : i32
      linalg.yield %val5 : i32
    } -> tensor<?x?x?x?x?x?xi32>
    return %res : tensor<?x?x?x?x?x?xi32>
  }
  func.func @sequential_rank_6_i1(%dim0: index, %dim1: index, %dim2: index, %dim3: index, %dim4: index, %dim5: index) -> tensor<?x?x?x?x?x?xi1> {
    %alloc = tensor.empty(%dim0, %dim1, %dim2, %dim3, %dim4, %dim5) : tensor<?x?x?x?x?x?xi1>
    %res = linalg.generic {
      indexing_maps = [#map6],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]
    } outs(%alloc : tensor<?x?x?x?x?x?xi1>) {
    ^bb0(%out: i1):
      %idx0 = linalg.index 0 : index
      %idx0_i32 = arith.index_cast %idx0 : index to i32
      %idx1 = linalg.index 1 : index
      %idx1_i32 = arith.index_cast %idx1 : index to i32
      %idx2 = linalg.index 2 : index
      %idx2_i32 = arith.index_cast %idx2 : index to i32
      %idx3 = linalg.index 3 : index
      %idx3_i32 = arith.index_cast %idx3 : index to i32
      %idx4 = linalg.index 4 : index
      %idx4_i32 = arith.index_cast %idx4 : index to i32
      %idx5 = linalg.index 5 : index
      %idx5_i32 = arith.index_cast %idx5 : index to i32
      %dim1_i32 = arith.index_cast %dim1 : index to i32
      %tmp1 = arith.muli %idx0_i32, %dim1_i32 : i32
      %val1 = arith.addi %idx1_i32, %tmp1 : i32
      %dim2_i32 = arith.index_cast %dim2 : index to i32
      %tmp2 = arith.muli %val1, %dim2_i32 : i32
      %val2 = arith.addi %idx2_i32, %tmp2 : i32
      %dim3_i32 = arith.index_cast %dim3 : index to i32
      %tmp3 = arith.muli %val2, %dim3_i32 : i32
      %val3 = arith.addi %idx3_i32, %tmp3 : i32
      %dim4_i32 = arith.index_cast %dim4 : index to i32
      %tmp4 = arith.muli %val3, %dim4_i32 : i32
      %val4 = arith.addi %idx4_i32, %tmp4 : i32
      %dim5_i32 = arith.index_cast %dim5 : index to i32
      %tmp5 = arith.muli %val4, %dim5_i32 : i32
      %val5 = arith.addi %idx5_i32, %tmp5 : i32
      %val_i1 = arith.trunci %val5 : i32 to i1
      linalg.yield %val_i1 : i1
    } -> tensor<?x?x?x?x?x?xi1>
    return %res : tensor<?x?x?x?x?x?xi1>
  }
  func.func @sequential_rank_7_i32(%dim0: index, %dim1: index, %dim2: index, %dim3: index, %dim4: index, %dim5: index, %dim6: index) -> tensor<?x?x?x?x?x?x?xi32> {
    %alloc = tensor.empty(%dim0, %dim1, %dim2, %dim3, %dim4, %dim5, %dim6) : tensor<?x?x?x?x?x?x?xi32>
    %res = linalg.generic {
      indexing_maps = [#map7],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]
    } outs(%alloc : tensor<?x?x?x?x?x?x?xi32>) {
    ^bb0(%out: i32):
      %idx0 = linalg.index 0 : index
      %idx0_i32 = arith.index_cast %idx0 : index to i32
      %idx1 = linalg.index 1 : index
      %idx1_i32 = arith.index_cast %idx1 : index to i32
      %idx2 = linalg.index 2 : index
      %idx2_i32 = arith.index_cast %idx2 : index to i32
      %idx3 = linalg.index 3 : index
      %idx3_i32 = arith.index_cast %idx3 : index to i32
      %idx4 = linalg.index 4 : index
      %idx4_i32 = arith.index_cast %idx4 : index to i32
      %idx5 = linalg.index 5 : index
      %idx5_i32 = arith.index_cast %idx5 : index to i32
      %idx6 = linalg.index 6 : index
      %idx6_i32 = arith.index_cast %idx6 : index to i32
      %dim1_i32 = arith.index_cast %dim1 : index to i32
      %tmp1 = arith.muli %idx0_i32, %dim1_i32 : i32
      %val1 = arith.addi %idx1_i32, %tmp1 : i32
      %dim2_i32 = arith.index_cast %dim2 : index to i32
      %tmp2 = arith.muli %val1, %dim2_i32 : i32
      %val2 = arith.addi %idx2_i32, %tmp2 : i32
      %dim3_i32 = arith.index_cast %dim3 : index to i32
      %tmp3 = arith.muli %val2, %dim3_i32 : i32
      %val3 = arith.addi %idx3_i32, %tmp3 : i32
      %dim4_i32 = arith.index_cast %dim4 : index to i32
      %tmp4 = arith.muli %val3, %dim4_i32 : i32
      %val4 = arith.addi %idx4_i32, %tmp4 : i32
      %dim5_i32 = arith.index_cast %dim5 : index to i32
      %tmp5 = arith.muli %val4, %dim5_i32 : i32
      %val5 = arith.addi %idx5_i32, %tmp5 : i32
      %dim6_i32 = arith.index_cast %dim6 : index to i32
      %tmp6 = arith.muli %val5, %dim6_i32 : i32
      %val6 = arith.addi %idx6_i32, %tmp6 : i32
      linalg.yield %val6 : i32
    } -> tensor<?x?x?x?x?x?x?xi32>
    return %res : tensor<?x?x?x?x?x?x?xi32>
  }
  func.func @sequential_rank_7_i1(%dim0: index, %dim1: index, %dim2: index, %dim3: index, %dim4: index, %dim5: index, %dim6: index) -> tensor<?x?x?x?x?x?x?xi1> {
    %alloc = tensor.empty(%dim0, %dim1, %dim2, %dim3, %dim4, %dim5, %dim6) : tensor<?x?x?x?x?x?x?xi1>
    %res = linalg.generic {
      indexing_maps = [#map7],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]
    } outs(%alloc : tensor<?x?x?x?x?x?x?xi1>) {
    ^bb0(%out: i1):
      %idx0 = linalg.index 0 : index
      %idx0_i32 = arith.index_cast %idx0 : index to i32
      %idx1 = linalg.index 1 : index
      %idx1_i32 = arith.index_cast %idx1 : index to i32
      %idx2 = linalg.index 2 : index
      %idx2_i32 = arith.index_cast %idx2 : index to i32
      %idx3 = linalg.index 3 : index
      %idx3_i32 = arith.index_cast %idx3 : index to i32
      %idx4 = linalg.index 4 : index
      %idx4_i32 = arith.index_cast %idx4 : index to i32
      %idx5 = linalg.index 5 : index
      %idx5_i32 = arith.index_cast %idx5 : index to i32
      %idx6 = linalg.index 6 : index
      %idx6_i32 = arith.index_cast %idx6 : index to i32
      %dim1_i32 = arith.index_cast %dim1 : index to i32
      %tmp1 = arith.muli %idx0_i32, %dim1_i32 : i32
      %val1 = arith.addi %idx1_i32, %tmp1 : i32
      %dim2_i32 = arith.index_cast %dim2 : index to i32
      %tmp2 = arith.muli %val1, %dim2_i32 : i32
      %val2 = arith.addi %idx2_i32, %tmp2 : i32
      %dim3_i32 = arith.index_cast %dim3 : index to i32
      %tmp3 = arith.muli %val2, %dim3_i32 : i32
      %val3 = arith.addi %idx3_i32, %tmp3 : i32
      %dim4_i32 = arith.index_cast %dim4 : index to i32
      %tmp4 = arith.muli %val3, %dim4_i32 : i32
      %val4 = arith.addi %idx4_i32, %tmp4 : i32
      %dim5_i32 = arith.index_cast %dim5 : index to i32
      %tmp5 = arith.muli %val4, %dim5_i32 : i32
      %val5 = arith.addi %idx5_i32, %tmp5 : i32
      %dim6_i32 = arith.index_cast %dim6 : index to i32
      %tmp6 = arith.muli %val5, %dim6_i32 : i32
      %val6 = arith.addi %idx6_i32, %tmp6 : i32
      %val_i1 = arith.trunci %val6 : i32 to i1
      linalg.yield %val_i1 : i1
    } -> tensor<?x?x?x?x?x?x?xi1>
    return %res : tensor<?x?x?x?x?x?x?xi1>
  }
  func.func @sequential_rank_8_i32(%dim0: index, %dim1: index, %dim2: index, %dim3: index, %dim4: index, %dim5: index, %dim6: index, %dim7: index) -> tensor<?x?x?x?x?x?x?x?xi32> {
    %alloc = tensor.empty(%dim0, %dim1, %dim2, %dim3, %dim4, %dim5, %dim6, %dim7) : tensor<?x?x?x?x?x?x?x?xi32>
    %res = linalg.generic {
      indexing_maps = [#map8],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]
    } outs(%alloc : tensor<?x?x?x?x?x?x?x?xi32>) {
    ^bb0(%out: i32):
      %idx0 = linalg.index 0 : index
      %idx0_i32 = arith.index_cast %idx0 : index to i32
      %idx1 = linalg.index 1 : index
      %idx1_i32 = arith.index_cast %idx1 : index to i32
      %idx2 = linalg.index 2 : index
      %idx2_i32 = arith.index_cast %idx2 : index to i32
      %idx3 = linalg.index 3 : index
      %idx3_i32 = arith.index_cast %idx3 : index to i32
      %idx4 = linalg.index 4 : index
      %idx4_i32 = arith.index_cast %idx4 : index to i32
      %idx5 = linalg.index 5 : index
      %idx5_i32 = arith.index_cast %idx5 : index to i32
      %idx6 = linalg.index 6 : index
      %idx6_i32 = arith.index_cast %idx6 : index to i32
      %idx7 = linalg.index 7 : index
      %idx7_i32 = arith.index_cast %idx7 : index to i32
      %dim1_i32 = arith.index_cast %dim1 : index to i32
      %tmp1 = arith.muli %idx0_i32, %dim1_i32 : i32
      %val1 = arith.addi %idx1_i32, %tmp1 : i32
      %dim2_i32 = arith.index_cast %dim2 : index to i32
      %tmp2 = arith.muli %val1, %dim2_i32 : i32
      %val2 = arith.addi %idx2_i32, %tmp2 : i32
      %dim3_i32 = arith.index_cast %dim3 : index to i32
      %tmp3 = arith.muli %val2, %dim3_i32 : i32
      %val3 = arith.addi %idx3_i32, %tmp3 : i32
      %dim4_i32 = arith.index_cast %dim4 : index to i32
      %tmp4 = arith.muli %val3, %dim4_i32 : i32
      %val4 = arith.addi %idx4_i32, %tmp4 : i32
      %dim5_i32 = arith.index_cast %dim5 : index to i32
      %tmp5 = arith.muli %val4, %dim5_i32 : i32
      %val5 = arith.addi %idx5_i32, %tmp5 : i32
      %dim6_i32 = arith.index_cast %dim6 : index to i32
      %tmp6 = arith.muli %val5, %dim6_i32 : i32
      %val6 = arith.addi %idx6_i32, %tmp6 : i32
      %dim7_i32 = arith.index_cast %dim7 : index to i32
      %tmp7 = arith.muli %val6, %dim7_i32 : i32
      %val7 = arith.addi %idx7_i32, %tmp7 : i32
      linalg.yield %val7 : i32
    } -> tensor<?x?x?x?x?x?x?x?xi32>
    return %res : tensor<?x?x?x?x?x?x?x?xi32>
  }
  func.func @sequential_rank_8_i1(%dim0: index, %dim1: index, %dim2: index, %dim3: index, %dim4: index, %dim5: index, %dim6: index, %dim7: index) -> tensor<?x?x?x?x?x?x?x?xi1> {
    %alloc = tensor.empty(%dim0, %dim1, %dim2, %dim3, %dim4, %dim5, %dim6, %dim7) : tensor<?x?x?x?x?x?x?x?xi1>
    %res = linalg.generic {
      indexing_maps = [#map8],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]
    } outs(%alloc : tensor<?x?x?x?x?x?x?x?xi1>) {
    ^bb0(%out: i1):
      %idx0 = linalg.index 0 : index
      %idx0_i32 = arith.index_cast %idx0 : index to i32
      %idx1 = linalg.index 1 : index
      %idx1_i32 = arith.index_cast %idx1 : index to i32
      %idx2 = linalg.index 2 : index
      %idx2_i32 = arith.index_cast %idx2 : index to i32
      %idx3 = linalg.index 3 : index
      %idx3_i32 = arith.index_cast %idx3 : index to i32
      %idx4 = linalg.index 4 : index
      %idx4_i32 = arith.index_cast %idx4 : index to i32
      %idx5 = linalg.index 5 : index
      %idx5_i32 = arith.index_cast %idx5 : index to i32
      %idx6 = linalg.index 6 : index
      %idx6_i32 = arith.index_cast %idx6 : index to i32
      %idx7 = linalg.index 7 : index
      %idx7_i32 = arith.index_cast %idx7 : index to i32
      %dim1_i32 = arith.index_cast %dim1 : index to i32
      %tmp1 = arith.muli %idx0_i32, %dim1_i32 : i32
      %val1 = arith.addi %idx1_i32, %tmp1 : i32
      %dim2_i32 = arith.index_cast %dim2 : index to i32
      %tmp2 = arith.muli %val1, %dim2_i32 : i32
      %val2 = arith.addi %idx2_i32, %tmp2 : i32
      %dim3_i32 = arith.index_cast %dim3 : index to i32
      %tmp3 = arith.muli %val2, %dim3_i32 : i32
      %val3 = arith.addi %idx3_i32, %tmp3 : i32
      %dim4_i32 = arith.index_cast %dim4 : index to i32
      %tmp4 = arith.muli %val3, %dim4_i32 : i32
      %val4 = arith.addi %idx4_i32, %tmp4 : i32
      %dim5_i32 = arith.index_cast %dim5 : index to i32
      %tmp5 = arith.muli %val4, %dim5_i32 : i32
      %val5 = arith.addi %idx5_i32, %tmp5 : i32
      %dim6_i32 = arith.index_cast %dim6 : index to i32
      %tmp6 = arith.muli %val5, %dim6_i32 : i32
      %val6 = arith.addi %idx6_i32, %tmp6 : i32
      %dim7_i32 = arith.index_cast %dim7 : index to i32
      %tmp7 = arith.muli %val6, %dim7_i32 : i32
      %val7 = arith.addi %idx7_i32, %tmp7 : i32
      %val_i1 = arith.trunci %val7 : i32 to i1
      linalg.yield %val_i1 : i1
    } -> tensor<?x?x?x?x?x?x?x?xi1>
    return %res : tensor<?x?x?x?x?x?x?x?xi1>
  }
  func.func @sequential_rank_9_i32(%dim0: index, %dim1: index, %dim2: index, %dim3: index, %dim4: index, %dim5: index, %dim6: index, %dim7: index, %dim8: index) -> tensor<?x?x?x?x?x?x?x?x?xi32> {
    %alloc = tensor.empty(%dim0, %dim1, %dim2, %dim3, %dim4, %dim5, %dim6, %dim7, %dim8) : tensor<?x?x?x?x?x?x?x?x?xi32>
    %res = linalg.generic {
      indexing_maps = [#map9],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]
    } outs(%alloc : tensor<?x?x?x?x?x?x?x?x?xi32>) {
    ^bb0(%out: i32):
      %idx0 = linalg.index 0 : index
      %idx0_i32 = arith.index_cast %idx0 : index to i32
      %idx1 = linalg.index 1 : index
      %idx1_i32 = arith.index_cast %idx1 : index to i32
      %idx2 = linalg.index 2 : index
      %idx2_i32 = arith.index_cast %idx2 : index to i32
      %idx3 = linalg.index 3 : index
      %idx3_i32 = arith.index_cast %idx3 : index to i32
      %idx4 = linalg.index 4 : index
      %idx4_i32 = arith.index_cast %idx4 : index to i32
      %idx5 = linalg.index 5 : index
      %idx5_i32 = arith.index_cast %idx5 : index to i32
      %idx6 = linalg.index 6 : index
      %idx6_i32 = arith.index_cast %idx6 : index to i32
      %idx7 = linalg.index 7 : index
      %idx7_i32 = arith.index_cast %idx7 : index to i32
      %idx8 = linalg.index 8 : index
      %idx8_i32 = arith.index_cast %idx8 : index to i32
      %dim1_i32 = arith.index_cast %dim1 : index to i32
      %tmp1 = arith.muli %idx0_i32, %dim1_i32 : i32
      %val1 = arith.addi %idx1_i32, %tmp1 : i32
      %dim2_i32 = arith.index_cast %dim2 : index to i32
      %tmp2 = arith.muli %val1, %dim2_i32 : i32
      %val2 = arith.addi %idx2_i32, %tmp2 : i32
      %dim3_i32 = arith.index_cast %dim3 : index to i32
      %tmp3 = arith.muli %val2, %dim3_i32 : i32
      %val3 = arith.addi %idx3_i32, %tmp3 : i32
      %dim4_i32 = arith.index_cast %dim4 : index to i32
      %tmp4 = arith.muli %val3, %dim4_i32 : i32
      %val4 = arith.addi %idx4_i32, %tmp4 : i32
      %dim5_i32 = arith.index_cast %dim5 : index to i32
      %tmp5 = arith.muli %val4, %dim5_i32 : i32
      %val5 = arith.addi %idx5_i32, %tmp5 : i32
      %dim6_i32 = arith.index_cast %dim6 : index to i32
      %tmp6 = arith.muli %val5, %dim6_i32 : i32
      %val6 = arith.addi %idx6_i32, %tmp6 : i32
      %dim7_i32 = arith.index_cast %dim7 : index to i32
      %tmp7 = arith.muli %val6, %dim7_i32 : i32
      %val7 = arith.addi %idx7_i32, %tmp7 : i32
      %dim8_i32 = arith.index_cast %dim8 : index to i32
      %tmp8 = arith.muli %val7, %dim8_i32 : i32
      %val8 = arith.addi %idx8_i32, %tmp8 : i32
      linalg.yield %val8 : i32
    } -> tensor<?x?x?x?x?x?x?x?x?xi32>
    return %res : tensor<?x?x?x?x?x?x?x?x?xi32>
  }
  func.func @sequential_rank_9_i1(%dim0: index, %dim1: index, %dim2: index, %dim3: index, %dim4: index, %dim5: index, %dim6: index, %dim7: index, %dim8: index) -> tensor<?x?x?x?x?x?x?x?x?xi1> {
    %alloc = tensor.empty(%dim0, %dim1, %dim2, %dim3, %dim4, %dim5, %dim6, %dim7, %dim8) : tensor<?x?x?x?x?x?x?x?x?xi1>
    %res = linalg.generic {
      indexing_maps = [#map9],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]
    } outs(%alloc : tensor<?x?x?x?x?x?x?x?x?xi1>) {
    ^bb0(%out: i1):
      %idx0 = linalg.index 0 : index
      %idx0_i32 = arith.index_cast %idx0 : index to i32
      %idx1 = linalg.index 1 : index
      %idx1_i32 = arith.index_cast %idx1 : index to i32
      %idx2 = linalg.index 2 : index
      %idx2_i32 = arith.index_cast %idx2 : index to i32
      %idx3 = linalg.index 3 : index
      %idx3_i32 = arith.index_cast %idx3 : index to i32
      %idx4 = linalg.index 4 : index
      %idx4_i32 = arith.index_cast %idx4 : index to i32
      %idx5 = linalg.index 5 : index
      %idx5_i32 = arith.index_cast %idx5 : index to i32
      %idx6 = linalg.index 6 : index
      %idx6_i32 = arith.index_cast %idx6 : index to i32
      %idx7 = linalg.index 7 : index
      %idx7_i32 = arith.index_cast %idx7 : index to i32
      %idx8 = linalg.index 8 : index
      %idx8_i32 = arith.index_cast %idx8 : index to i32
      %dim1_i32 = arith.index_cast %dim1 : index to i32
      %tmp1 = arith.muli %idx0_i32, %dim1_i32 : i32
      %val1 = arith.addi %idx1_i32, %tmp1 : i32
      %dim2_i32 = arith.index_cast %dim2 : index to i32
      %tmp2 = arith.muli %val1, %dim2_i32 : i32
      %val2 = arith.addi %idx2_i32, %tmp2 : i32
      %dim3_i32 = arith.index_cast %dim3 : index to i32
      %tmp3 = arith.muli %val2, %dim3_i32 : i32
      %val3 = arith.addi %idx3_i32, %tmp3 : i32
      %dim4_i32 = arith.index_cast %dim4 : index to i32
      %tmp4 = arith.muli %val3, %dim4_i32 : i32
      %val4 = arith.addi %idx4_i32, %tmp4 : i32
      %dim5_i32 = arith.index_cast %dim5 : index to i32
      %tmp5 = arith.muli %val4, %dim5_i32 : i32
      %val5 = arith.addi %idx5_i32, %tmp5 : i32
      %dim6_i32 = arith.index_cast %dim6 : index to i32
      %tmp6 = arith.muli %val5, %dim6_i32 : i32
      %val6 = arith.addi %idx6_i32, %tmp6 : i32
      %dim7_i32 = arith.index_cast %dim7 : index to i32
      %tmp7 = arith.muli %val6, %dim7_i32 : i32
      %val7 = arith.addi %idx7_i32, %tmp7 : i32
      %dim8_i32 = arith.index_cast %dim8 : index to i32
      %tmp8 = arith.muli %val7, %dim8_i32 : i32
      %val8 = arith.addi %idx8_i32, %tmp8 : i32
      %val_i1 = arith.trunci %val8 : i32 to i1
      linalg.yield %val_i1 : i1
    } -> tensor<?x?x?x?x?x?x?x?x?xi1>
    return %res : tensor<?x?x?x?x?x?x?x?x?xi1>
  }
  func.func @sequential_rank_10_i32(%dim0: index, %dim1: index, %dim2: index, %dim3: index, %dim4: index, %dim5: index, %dim6: index, %dim7: index, %dim8: index, %dim9: index) -> tensor<?x?x?x?x?x?x?x?x?x?xi32> {
    %alloc = tensor.empty(%dim0, %dim1, %dim2, %dim3, %dim4, %dim5, %dim6, %dim7, %dim8, %dim9) : tensor<?x?x?x?x?x?x?x?x?x?xi32>
    %res = linalg.generic {
      indexing_maps = [#map10],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]
    } outs(%alloc : tensor<?x?x?x?x?x?x?x?x?x?xi32>) {
    ^bb0(%out: i32):
      %idx0 = linalg.index 0 : index
      %idx0_i32 = arith.index_cast %idx0 : index to i32
      %idx1 = linalg.index 1 : index
      %idx1_i32 = arith.index_cast %idx1 : index to i32
      %idx2 = linalg.index 2 : index
      %idx2_i32 = arith.index_cast %idx2 : index to i32
      %idx3 = linalg.index 3 : index
      %idx3_i32 = arith.index_cast %idx3 : index to i32
      %idx4 = linalg.index 4 : index
      %idx4_i32 = arith.index_cast %idx4 : index to i32
      %idx5 = linalg.index 5 : index
      %idx5_i32 = arith.index_cast %idx5 : index to i32
      %idx6 = linalg.index 6 : index
      %idx6_i32 = arith.index_cast %idx6 : index to i32
      %idx7 = linalg.index 7 : index
      %idx7_i32 = arith.index_cast %idx7 : index to i32
      %idx8 = linalg.index 8 : index
      %idx8_i32 = arith.index_cast %idx8 : index to i32
      %idx9 = linalg.index 9 : index
      %idx9_i32 = arith.index_cast %idx9 : index to i32
      %dim1_i32 = arith.index_cast %dim1 : index to i32
      %tmp1 = arith.muli %idx0_i32, %dim1_i32 : i32
      %val1 = arith.addi %idx1_i32, %tmp1 : i32
      %dim2_i32 = arith.index_cast %dim2 : index to i32
      %tmp2 = arith.muli %val1, %dim2_i32 : i32
      %val2 = arith.addi %idx2_i32, %tmp2 : i32
      %dim3_i32 = arith.index_cast %dim3 : index to i32
      %tmp3 = arith.muli %val2, %dim3_i32 : i32
      %val3 = arith.addi %idx3_i32, %tmp3 : i32
      %dim4_i32 = arith.index_cast %dim4 : index to i32
      %tmp4 = arith.muli %val3, %dim4_i32 : i32
      %val4 = arith.addi %idx4_i32, %tmp4 : i32
      %dim5_i32 = arith.index_cast %dim5 : index to i32
      %tmp5 = arith.muli %val4, %dim5_i32 : i32
      %val5 = arith.addi %idx5_i32, %tmp5 : i32
      %dim6_i32 = arith.index_cast %dim6 : index to i32
      %tmp6 = arith.muli %val5, %dim6_i32 : i32
      %val6 = arith.addi %idx6_i32, %tmp6 : i32
      %dim7_i32 = arith.index_cast %dim7 : index to i32
      %tmp7 = arith.muli %val6, %dim7_i32 : i32
      %val7 = arith.addi %idx7_i32, %tmp7 : i32
      %dim8_i32 = arith.index_cast %dim8 : index to i32
      %tmp8 = arith.muli %val7, %dim8_i32 : i32
      %val8 = arith.addi %idx8_i32, %tmp8 : i32
      %dim9_i32 = arith.index_cast %dim9 : index to i32
      %tmp9 = arith.muli %val8, %dim9_i32 : i32
      %val9 = arith.addi %idx9_i32, %tmp9 : i32
      linalg.yield %val9 : i32
    } -> tensor<?x?x?x?x?x?x?x?x?x?xi32>
    return %res : tensor<?x?x?x?x?x?x?x?x?x?xi32>
  }
  func.func @sequential_rank_10_i1(%dim0: index, %dim1: index, %dim2: index, %dim3: index, %dim4: index, %dim5: index, %dim6: index, %dim7: index, %dim8: index, %dim9: index) -> tensor<?x?x?x?x?x?x?x?x?x?xi1> {
    %alloc = tensor.empty(%dim0, %dim1, %dim2, %dim3, %dim4, %dim5, %dim6, %dim7, %dim8, %dim9) : tensor<?x?x?x?x?x?x?x?x?x?xi1>
    %res = linalg.generic {
      indexing_maps = [#map10],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]
    } outs(%alloc : tensor<?x?x?x?x?x?x?x?x?x?xi1>) {
    ^bb0(%out: i1):
      %idx0 = linalg.index 0 : index
      %idx0_i32 = arith.index_cast %idx0 : index to i32
      %idx1 = linalg.index 1 : index
      %idx1_i32 = arith.index_cast %idx1 : index to i32
      %idx2 = linalg.index 2 : index
      %idx2_i32 = arith.index_cast %idx2 : index to i32
      %idx3 = linalg.index 3 : index
      %idx3_i32 = arith.index_cast %idx3 : index to i32
      %idx4 = linalg.index 4 : index
      %idx4_i32 = arith.index_cast %idx4 : index to i32
      %idx5 = linalg.index 5 : index
      %idx5_i32 = arith.index_cast %idx5 : index to i32
      %idx6 = linalg.index 6 : index
      %idx6_i32 = arith.index_cast %idx6 : index to i32
      %idx7 = linalg.index 7 : index
      %idx7_i32 = arith.index_cast %idx7 : index to i32
      %idx8 = linalg.index 8 : index
      %idx8_i32 = arith.index_cast %idx8 : index to i32
      %idx9 = linalg.index 9 : index
      %idx9_i32 = arith.index_cast %idx9 : index to i32
      %dim1_i32 = arith.index_cast %dim1 : index to i32
      %tmp1 = arith.muli %idx0_i32, %dim1_i32 : i32
      %val1 = arith.addi %idx1_i32, %tmp1 : i32
      %dim2_i32 = arith.index_cast %dim2 : index to i32
      %tmp2 = arith.muli %val1, %dim2_i32 : i32
      %val2 = arith.addi %idx2_i32, %tmp2 : i32
      %dim3_i32 = arith.index_cast %dim3 : index to i32
      %tmp3 = arith.muli %val2, %dim3_i32 : i32
      %val3 = arith.addi %idx3_i32, %tmp3 : i32
      %dim4_i32 = arith.index_cast %dim4 : index to i32
      %tmp4 = arith.muli %val3, %dim4_i32 : i32
      %val4 = arith.addi %idx4_i32, %tmp4 : i32
      %dim5_i32 = arith.index_cast %dim5 : index to i32
      %tmp5 = arith.muli %val4, %dim5_i32 : i32
      %val5 = arith.addi %idx5_i32, %tmp5 : i32
      %dim6_i32 = arith.index_cast %dim6 : index to i32
      %tmp6 = arith.muli %val5, %dim6_i32 : i32
      %val6 = arith.addi %idx6_i32, %tmp6 : i32
      %dim7_i32 = arith.index_cast %dim7 : index to i32
      %tmp7 = arith.muli %val6, %dim7_i32 : i32
      %val7 = arith.addi %idx7_i32, %tmp7 : i32
      %dim8_i32 = arith.index_cast %dim8 : index to i32
      %tmp8 = arith.muli %val7, %dim8_i32 : i32
      %val8 = arith.addi %idx8_i32, %tmp8 : i32
      %dim9_i32 = arith.index_cast %dim9 : index to i32
      %tmp9 = arith.muli %val8, %dim9_i32 : i32
      %val9 = arith.addi %idx9_i32, %tmp9 : i32
      %val_i1 = arith.trunci %val9 : i32 to i1
      linalg.yield %val_i1 : i1
    } -> tensor<?x?x?x?x?x?x?x?x?x?xi1>
    return %res : tensor<?x?x?x?x?x?x?x?x?x?xi1>
  }
}
