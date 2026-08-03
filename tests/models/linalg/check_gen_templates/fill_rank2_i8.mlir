func.func @main(%arg0: tensor<?x?xi8>) -> tensor<?x?xi8> {
  %cst = arith.constant 1 : i8
  %0 = linalg.fill ins(%cst : i8) outs(%arg0 : tensor<?x?xi8>) -> tensor<?x?xi8>
  return %0 : tensor<?x?xi8>
}
