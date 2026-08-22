// RUN: %template_path
// TODO: Enable this test when compiler crash is fixed. Upstream MLIR
// linalg.winograd_filter_transform crashes in IREE CPU fallback because it
// lacks PartitionableLoopsInterface (which IREE's LinalgExt variant has).

func.func @main(%arg0: tensor<?x3x3x?xbf16>) -> tensor<4x4x?x?xbf16> {
  %c0 = arith.constant 0.0 : bf16
  %c0_idx = arith.constant 0 : index
  %c3_idx = arith.constant 3 : index

  %f = tensor.dim %arg0, %c0_idx : tensor<?x3x3x?xbf16>
  %c = tensor.dim %arg0, %c3_idx : tensor<?x3x3x?xbf16>

  %empty = tensor.empty(%c, %f) : tensor<4x4x?x?xbf16>
  %fill = linalg.fill ins(%c0 : bf16) outs(%empty : tensor<4x4x?x?xbf16>) -> tensor<4x4x?x?xbf16>

  %0 = linalg.winograd_filter_transform fmr(F_2_3)
       ins(%arg0 : tensor<?x3x3x?xbf16>)
       outs(%fill : tensor<4x4x?x?xbf16>) -> tensor<4x4x?x?xbf16>
  return %0 : tensor<4x4x?x?xbf16>
}
