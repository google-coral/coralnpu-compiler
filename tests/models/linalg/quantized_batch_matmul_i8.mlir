func.func @main(%arg0: tensor<?x?x?xi8>, %arg1: tensor<?x?x?xi8>) -> tensor<?x?x?xi32> {
  %c0 = arith.constant 0 : i32
  %zp0 = arith.constant 2 : i32
  %zp1 = arith.constant -3 : i32

  %c0_idx = arith.constant 0 : index
  %c1_idx = arith.constant 1 : index
  %c2_idx = arith.constant 2 : index

  %batch = tensor.dim %arg0, %c0_idx : tensor<?x?x?xi8>
  %m = tensor.dim %arg0, %c1_idx : tensor<?x?x?xi8>
  %n = tensor.dim %arg1, %c2_idx : tensor<?x?x?xi8>

  %empty = tensor.empty(%batch, %m, %n) : tensor<?x?x?xi32>
  %fill = linalg.fill ins(%c0 : i32) outs(%empty : tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  %0 = linalg.quantized_batch_matmul
       ins(%arg0, %arg1, %zp0, %zp1 : tensor<?x?x?xi8>, tensor<?x?x?xi8>, i32, i32)
       outs(%fill : tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  return %0 : tensor<?x?x?xi32>
}
