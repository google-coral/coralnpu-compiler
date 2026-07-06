func.func @main(%arg0: tensor<?x?xi8>, %arg1: tensor<?x?xi8>) -> tensor<?x?xi8> {
  %0 = stablehlo.shift_right_arithmetic %arg0, %arg1 : (tensor<?x?xi8>, tensor<?x?xi8>) -> tensor<?x?xi8>
  return %0 : tensor<?x?xi8>
}
