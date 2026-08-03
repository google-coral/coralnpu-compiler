func.func @main(%arg0: tensor<?x?xi8>) -> tensor<?x?xi8> {
  %0 = stablehlo.abs %arg0 : tensor<?x?xi8>
  return %0 : tensor<?x?xi8>
}
