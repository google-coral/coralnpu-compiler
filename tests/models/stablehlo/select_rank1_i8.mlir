func.func @main(%pred: tensor<?xi1>, %on_true: tensor<?xi8>, %on_false: tensor<?xi8>) -> tensor<?xi8> {
  %0 = stablehlo.select %pred, %on_true, %on_false : (tensor<?xi1>, tensor<?xi8>, tensor<?xi8>) -> tensor<?xi8>
  return %0 : tensor<?xi8>
}
