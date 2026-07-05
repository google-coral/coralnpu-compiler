func.func @main(%arg0: tensor<?x?x?x?x?xi16>, %arg1: tensor<?x?x?x?x?xi16>) -> tensor<?x?x?x?x?xi16> {
  %c0 = arith.constant 0 : i16
  %c0_idx = arith.constant 0 : index
  %c1_idx = arith.constant 1 : index
  %c2_idx = arith.constant 2 : index
  %c3_idx = arith.constant 3 : index

  %batch = tensor.dim %arg0, %c0_idx : tensor<?x?x?x?x?xi16>
  %m1 = tensor.dim %arg0, %c1_idx : tensor<?x?x?x?x?xi16>
  %n1 = tensor.dim %arg1, %c1_idx : tensor<?x?x?x?x?xi16>
  %m0 = tensor.dim %arg0, %c3_idx : tensor<?x?x?x?x?xi16>
  %n0 = tensor.dim %arg1, %c3_idx : tensor<?x?x?x?x?xi16>

  %empty = tensor.empty(%batch, %m1, %n1, %m0, %n0) : tensor<?x?x?x?x?xi16>
  %fill = linalg.fill ins(%c0 : i16) outs(%empty : tensor<?x?x?x?x?xi16>) -> tensor<?x?x?x?x?xi16>
  %0 = linalg.batch_mmt4d
       ins(%arg0, %arg1 : tensor<?x?x?x?x?xi16>, tensor<?x?x?x?x?xi16>)
       outs(%fill : tensor<?x?x?x?x?xi16>) -> tensor<?x?x?x?x?xi16>
  return %0 : tensor<?x?x?x?x?xi16>
}
