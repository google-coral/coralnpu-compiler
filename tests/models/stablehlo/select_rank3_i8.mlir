func.func @main(%pred: tensor<?x?x?xi1>, %on_true: tensor<?x?x?xi8>, %on_false: tensor<?x?x?xi8>) -> tensor<?x?x?xi8> {
  %0 = stablehlo.select %pred, %on_true, %on_false : (tensor<?x?x?xi1>, tensor<?x?x?xi8>, tensor<?x?x?xi8>) -> tensor<?x?x?xi8>
  return %0 : tensor<?x?x?xi8>
}
