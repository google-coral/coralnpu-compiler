func.func @main(%index_tensor: tensor<1xi32>, %arg0: tensor<?xf32>) -> tensor<?xf32> {
  %index = stablehlo.reshape %index_tensor : (tensor<1xi32>) -> tensor<i32>
  %0 = "stablehlo.case"(%index) ({
    "stablehlo.return"(%arg0) : (tensor<?xf32>) -> ()
  }, {
    %neg = stablehlo.negate %arg0 : tensor<?xf32>
    "stablehlo.return"(%neg) : (tensor<?xf32>) -> ()
  }) : (tensor<i32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}
