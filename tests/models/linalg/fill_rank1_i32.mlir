func.func @main(%arg0: tensor<?xi32>) -> tensor<?xi32> {
  %cst = arith.constant 1 : i32
  %0 = linalg.fill ins(%cst : i32) outs(%arg0 : tensor<?xi32>) -> tensor<?xi32>
  return %0 : tensor<?xi32>
}
