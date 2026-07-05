func.func @main(%arg0: tensor<?xi16>, %arg1: tensor<?x?xi16>) -> tensor<?xi16> {
  %c0 = arith.constant 0 : i16
  %c1_idx = arith.constant 1 : index
  %n = tensor.dim %arg1, %c1_idx : tensor<?x?xi16>
  %empty = tensor.empty(%n) : tensor<?xi16>
  %fill = linalg.fill ins(%c0 : i16) outs(%empty : tensor<?xi16>) -> tensor<?xi16>
  %0 = linalg.vecmat
       ins(%arg0, %arg1 : tensor<?xi16>, tensor<?x?xi16>)
       outs(%fill : tensor<?xi16>) -> tensor<?xi16>
  return %0 : tensor<?xi16>
}
