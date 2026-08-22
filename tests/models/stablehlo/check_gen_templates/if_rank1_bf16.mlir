func.func @main(%pred_tensor: tensor<1xi1>, %arg0: tensor<?xbf16>) -> tensor<?xbf16> {
  %pred = stablehlo.reshape %pred_tensor : (tensor<1xi1>) -> tensor<i1>
  %0 = "stablehlo.if"(%pred) ({
    "stablehlo.return"(%arg0) : (tensor<?xbf16>) -> ()
  }, {
    %neg = stablehlo.negate %arg0 : tensor<?xbf16>
    "stablehlo.return"(%neg) : (tensor<?xbf16>) -> ()
  }) : (tensor<i1>) -> tensor<?xbf16>
  return %0 : tensor<?xbf16>
}
