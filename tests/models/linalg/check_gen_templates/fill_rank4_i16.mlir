func.func @main(%arg0: tensor<?x?x?x?xi16>) -> tensor<?x?x?x?xi16> {
  %cst = arith.constant 1 : i16
  %0 = linalg.fill ins(%cst : i16) outs(%arg0 : tensor<?x?x?x?xi16>) -> tensor<?x?x?x?xi16>
  return %0 : tensor<?x?x?x?xi16>
}
