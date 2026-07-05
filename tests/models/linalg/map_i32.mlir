func.func @main(%arg0: tensor<?x?xi32>, %arg1: tensor<?x?xi32>) -> tensor<?x?xi32> {
  %c0_idx = arith.constant 0 : index
  %c1_idx = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0_idx : tensor<?x?xi32>
  %d1 = tensor.dim %arg0, %c1_idx : tensor<?x?xi32>

  %empty = tensor.empty(%d0, %d1) : tensor<?x?xi32>
  %0 = linalg.map ins(%arg0, %arg1 : tensor<?x?xi32>, tensor<?x?xi32>) outs(%empty : tensor<?x?xi32>) (%val0: i32, %val1: i32, %out: i32) {
    %res = arith.addi %val0, %val1 : i32
    linalg.yield %res : i32
  }
  return %0 : tensor<?x?xi32>
}
