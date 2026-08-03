#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @index_i32_4_8() {
    %cst = arith.constant dense<[[0, 1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7, 8], [2, 3, 4, 5, 6, 7, 8, 9], [3, 4, 5, 6, 7, 8, 9, 10]]> : tensor<4x8xi32>
    %cst_0 = arith.constant dense<[[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15], [16, 17, 18, 19, 20, 21, 22, 23], [24, 25, 26, 27, 28, 29, 30, 31]]> : tensor<4x8xi32>
    %0 = util.optimization_barrier %cst_0 : tensor<4x8xi32>
    %1 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%0 : tensor<4x8xi32>) {
    ^bb0(%out: i32):
      %3 = linalg.index 0 : index
      %4 = linalg.index 1 : index
      %5 = arith.index_cast %3 : index to i32
      %6 = arith.index_cast %4 : index to i32
      %7 = arith.addi %5, %6 : i32
      linalg.yield %7 : i32
    } -> tensor<4x8xi32>
    %2 = util.optimization_barrier %cst : tensor<4x8xi32>
    "check.expect_eq"(%1, %2) : (tensor<4x8xi32>, tensor<4x8xi32>) -> ()
    return
  }
}
