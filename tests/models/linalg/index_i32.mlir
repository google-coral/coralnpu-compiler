// RUN: %template_path

func.func @main(%out: tensor<?x?xi32>) -> tensor<?x?xi32> {
  %res = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } outs(%out : tensor<?x?xi32>) {
  ^bb0(%out_val: i32):
    %row = linalg.index 0 : index
    %col = linalg.index 1 : index
    %row_i32 = arith.index_cast %row : index to i32
    %col_i32 = arith.index_cast %col : index to i32
    %val = arith.addi %row_i32, %col_i32 : i32
    linalg.yield %val : i32
  } -> tensor<?x?xi32>
  return %res : tensor<?x?xi32>
}
