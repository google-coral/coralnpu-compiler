func.func @main(%arg0: tensor<?x?x?xi32>) -> tensor<?x?x?xi32> {
  %cst = arith.constant 1 : i32
  %0 = linalg.fill ins(%cst : i32) outs(%arg0 : tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  return %0 : tensor<?x?x?xi32>
}
