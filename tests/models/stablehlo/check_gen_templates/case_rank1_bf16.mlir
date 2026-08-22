func.func @main(%index_tensor: tensor<1xi32>, %arg0: tensor<?xbf16>) -> tensor<?xbf16> {
  %index = stablehlo.reshape %index_tensor : (tensor<1xi32>) -> tensor<i32>
  %0 = "stablehlo.case"(%index) ({
    "stablehlo.return"(%arg0) : (tensor<?xbf16>) -> ()
  }, {
    %neg = stablehlo.negate %arg0 : tensor<?xbf16>
    "stablehlo.return"(%neg) : (tensor<?xbf16>) -> ()
  }) : (tensor<i32>) -> tensor<?xbf16>
  return %0 : tensor<?xbf16>
}
