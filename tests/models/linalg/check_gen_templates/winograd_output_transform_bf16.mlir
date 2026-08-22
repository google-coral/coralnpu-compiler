// RUN: %template_path
// TODO: Enable this test when compiler crash is fixed. Upstream MLIR
// linalg.winograd_output_transform crashes in IREE CPU fallback because it
// lacks PartitionableLoopsInterface (which IREE's LinalgExt variant has).

func.func @main(%arg0: tensor<4x4x2x2x1x8xbf16>, %arg1: tensor<1x4x4x8xbf16>) -> tensor<1x4x4x8xbf16> {
  %0 = linalg.winograd_output_transform fmr(F_2_3)
       ins(%arg0 : tensor<4x4x2x2x1x8xbf16>)
       outs(%arg1 : tensor<1x4x4x8xbf16>) -> tensor<1x4x4x8xbf16>
  return %0 : tensor<1x4x4x8xbf16>
}
