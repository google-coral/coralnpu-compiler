func.func @main(%arg0: tensor<?x?x?xi8>, %arg1: tensor<?x?x?xi8>) -> tensor<?x?x?xi8> {
  %0 = stablehlo.power %arg0, %arg1 : (tensor<?x?x?xi8>, tensor<?x?x?xi8>) -> tensor<?x?x?xi8>
  return %0 : tensor<?x?x?xi8>
}
