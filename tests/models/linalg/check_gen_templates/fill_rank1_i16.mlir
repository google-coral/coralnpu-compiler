func.func @main(%arg0: tensor<?xi16>) -> tensor<?xi16> {
  %cst = arith.constant 1 : i16
  %0 = linalg.fill ins(%cst : i16) outs(%arg0 : tensor<?xi16>) -> tensor<?xi16>
  return %0 : tensor<?xi16>
}
