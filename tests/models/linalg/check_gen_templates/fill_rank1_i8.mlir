func.func @main(%arg0: tensor<?xi8>) -> tensor<?xi8> {
  %cst = arith.constant 1 : i8
  %0 = linalg.fill ins(%cst : i8) outs(%arg0 : tensor<?xi8>) -> tensor<?xi8>
  return %0 : tensor<?xi8>
}
