#mapA = affine_map<(m, n, k) -> (m, k)>
#mapB = affine_map<(m, n, k) -> (k, n)>
#mapC = affine_map<(m, n, k) -> (m, n)>

func.func @main(%arg0: tensor<?x?xi32>, %arg1: tensor<?x?xi32>) -> tensor<?x?xi32> {
  %c0 = arith.constant 0 : i32
  %c0_idx = arith.constant 0 : index
  %c1_idx = arith.constant 1 : index
  %m = tensor.dim %arg0, %c0_idx : tensor<?x?xi32>
  %n = tensor.dim %arg1, %c1_idx : tensor<?x?xi32>

  %empty = tensor.empty(%m, %n) : tensor<?x?xi32>
  %fill = linalg.fill ins(%c0 : i32) outs(%empty : tensor<?x?xi32>) -> tensor<?x?xi32>
  %0 = linalg.contract
      indexing_maps = [#mapA, #mapB, #mapC]
      ins(%arg0, %arg1 : tensor<?x?xi32>, tensor<?x?xi32>)
      outs(%fill : tensor<?x?xi32>) -> tensor<?x?xi32>
  return %0 : tensor<?x?xi32>
}
