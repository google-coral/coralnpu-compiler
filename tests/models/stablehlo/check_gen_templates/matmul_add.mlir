func.func @main(
    %arg0: tensor<4x8xi32>,
    %arg1: tensor<8x4xi32>,
    %arg2: tensor<4x4xi32>,
    %arg3: tensor<4x4xi32>)
    -> (tensor<4x4xi32>, tensor<4x4xi32>) {
  %matmul = stablehlo.dot %arg0, %arg1
      // {stream.affinity = #hal.device.promise<@npu_device>}
      : (tensor<4x8xi32>, tensor<8x4xi32>) -> tensor<4x4xi32>

  %add = stablehlo.add %arg2, %arg3 : tensor<4x4xi32>

  return %matmul, %add : tensor<4x4xi32>, tensor<4x4xi32>
}
