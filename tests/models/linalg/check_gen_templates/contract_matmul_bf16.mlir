#mapA = affine_map<(m, n, k) -> (m, k)>
#mapB = affine_map<(m, n, k) -> (k, n)>
#mapC = affine_map<(m, n, k) -> (m, n)>

func.func @main(%arg0: tensor<?x?xbf16>, %arg1: tensor<?x?xbf16>) -> tensor<?x?xbf16> {
  %c0 = arith.constant 0.0 : bf16
  %c0_idx = arith.constant 0 : index
  %c1_idx = arith.constant 1 : index
  %m = tensor.dim %arg0, %c0_idx : tensor<?x?xbf16>
  %n = tensor.dim %arg1, %c1_idx : tensor<?x?xbf16>

  %empty = tensor.empty(%m, %n) : tensor<?x?xbf16>
  %fill = linalg.fill ins(%c0 : bf16) outs(%empty : tensor<?x?xbf16>) -> tensor<?x?xbf16>
  %0 = linalg.contract
      indexing_maps = [#mapA, #mapB, #mapC]
      ins(%arg0, %arg1 : tensor<?x?xbf16>, tensor<?x?xbf16>)
      outs(%fill : tensor<?x?xbf16>) -> tensor<?x?xbf16>
  return %0 : tensor<?x?xbf16>
}
