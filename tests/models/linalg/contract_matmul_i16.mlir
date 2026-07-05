#mapA = affine_map<(m, n, k) -> (m, k)>
#mapB = affine_map<(m, n, k) -> (k, n)>
#mapC = affine_map<(m, n, k) -> (m, n)>

func.func @main(%arg0: tensor<?x?xi16>, %arg1: tensor<?x?xi16>) -> tensor<?x?xi16> {
  %c0 = arith.constant 0 : i16
  %c0_idx = arith.constant 0 : index
  %c1_idx = arith.constant 1 : index
  %m = tensor.dim %arg0, %c0_idx : tensor<?x?xi16>
  %n = tensor.dim %arg1, %c1_idx : tensor<?x?xi16>

  %empty = tensor.empty(%m, %n) : tensor<?x?xi16>
  %fill = linalg.fill ins(%c0 : i16) outs(%empty : tensor<?x?xi16>) -> tensor<?x?xi16>
  %0 = linalg.contract
      indexing_maps = [#mapA, #mapB, #mapC]
      ins(%arg0, %arg1 : tensor<?x?xi16>, tensor<?x?xi16>)
      outs(%fill : tensor<?x?xi16>) -> tensor<?x?xi16>
  return %0 : tensor<?x?xi16>
}
