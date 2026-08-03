// RUN: %template_path
// TODO: Enable this test when compiler crash is fixed. Upstream MLIR
// linalg.winograd_output_transform crashes in IREE CPU fallback because it
// lacks PartitionableLoopsInterface (which IREE's LinalgExt variant has).

func.func @main(%arg0: tensor<4x4x2x2x1x8xf32>, %arg1: tensor<1x4x4x8xf32>) -> tensor<1x4x4x8xf32> {
  %0 = linalg.winograd_output_transform fmr(F_2_3)
       ins(%arg0 : tensor<4x4x2x2x1x8xf32>)
       outs(%arg1 : tensor<1x4x4x8xf32>) -> tensor<1x4x4x8xf32>
  return %0 : tensor<1x4x4x8xf32>
}
