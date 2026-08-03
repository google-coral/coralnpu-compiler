func.func @main(%pred_tensor: tensor<1xi1>, %arg0: tensor<?xf32>) -> tensor<?xf32> {
  %pred = stablehlo.reshape %pred_tensor : (tensor<1xi1>) -> tensor<i1>
  %0 = "stablehlo.if"(%pred) ({
    "stablehlo.return"(%arg0) : (tensor<?xf32>) -> ()
  }, {
    %neg = stablehlo.negate %arg0 : tensor<?xf32>
    "stablehlo.return"(%neg) : (tensor<?xf32>) -> ()
  }) : (tensor<i1>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}
