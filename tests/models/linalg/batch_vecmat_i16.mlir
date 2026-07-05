func.func @main(%arg0: tensor<?x?xi16>, %arg1: tensor<?x?x?xi16>) -> tensor<?x?xi16> {
  %c0 = arith.constant 0 : i16
  %c0_idx = arith.constant 0 : index
  %c2_idx = arith.constant 2 : index
  %batch = tensor.dim %arg1, %c0_idx : tensor<?x?x?xi16>
  %n = tensor.dim %arg1, %c2_idx : tensor<?x?x?xi16>
  %empty = tensor.empty(%batch, %n) : tensor<?x?xi16>
  %fill = linalg.fill ins(%c0 : i16) outs(%empty : tensor<?x?xi16>) -> tensor<?x?xi16>
  %0 = linalg.batch_vecmat
       ins(%arg0, %arg1 : tensor<?x?xi16>, tensor<?x?x?xi16>)
       outs(%fill : tensor<?x?xi16>) -> tensor<?x?xi16>
  return %0 : tensor<?x?xi16>
}
