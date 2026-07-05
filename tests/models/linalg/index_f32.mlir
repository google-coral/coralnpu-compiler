// RUN: %template_path

func.func @main(%out: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %res = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } outs(%out : tensor<?x?xf32>) {
  ^bb0(%out_val: f32):
    %row = linalg.index 0 : index
    %col = linalg.index 1 : index
    %row_i32 = arith.index_cast %row : index to i32
    %col_i32 = arith.index_cast %col : index to i32
    %row_f32 = arith.sitofp %row_i32 : i32 to f32
    %col_f32 = arith.sitofp %col_i32 : i32 to f32
    %val = arith.addf %row_f32, %col_f32 : f32
    linalg.yield %val : f32
  } -> tensor<?x?xf32>
  return %res : tensor<?x?xf32>
}
