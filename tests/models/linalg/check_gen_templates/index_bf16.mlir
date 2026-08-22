// RUN: %template_path

func.func @main(%out: tensor<?x?xbf16>) -> tensor<?x?xbf16> {
  %res = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } outs(%out : tensor<?x?xbf16>) {
  ^bb0(%out_val: bf16):
    %row = linalg.index 0 : index
    %col = linalg.index 1 : index
    %row_i32 = arith.index_cast %row : index to i32
    %col_i32 = arith.index_cast %col : index to i32
    %row_bf16 = arith.sitofp %row_i32 : i32 to bf16
    %col_bf16 = arith.sitofp %col_i32 : i32 to bf16
    %val = arith.addf %row_bf16, %col_bf16 : bf16
    linalg.yield %val : bf16
  } -> tensor<?x?xbf16>
  return %res : tensor<?x?xbf16>
}
