// RUN: iree-compile --mlir-disable-threading --iree-hal-target-device=coralnpu \
// RUN:   --compile-to=executable-configurations %s | FileCheck %s
// RUN: iree-compile --mlir-disable-threading --iree-hal-target-device=coralnpu \
// RUN:   --compile-to=executable-configurations \
// RUN:   --coralnpu-dtcm-buffers-per-operand=2 %s \
// RUN:   | FileCheck %s --check-prefix=DOUBLE
// RUN: iree-compile --mlir-disable-threading --iree-hal-target-device=coralnpu \
// RUN:   --compile-to=executable-configurations --coralnpu-dtcm-size-kb=12 %s \
// RUN:   2>&1 | FileCheck %s --check-prefix=SMALL --implicit-check-not=warning:
// RUN: iree-compile --mlir-disable-threading --iree-hal-target-device=coralnpu \
// RUN:   --compile-to=executable-configurations --coralnpu-dtcm-size-kb=1024 %s \
// RUN:   | FileCheck %s --check-prefix=HIGHMEM
// RUN: iree-compile --mlir-disable-threading --iree-hal-target-device=coralnpu \
// RUN:   --compile-to=executable-configurations \
// RUN:   --coralnpu-target-cpu-features=+m,+f,+zvl128b,+zve32f %s \
// RUN:   | FileCheck %s --check-prefix=NOZVT
// RUN: iree-compile --mlir-disable-threading --iree-hal-target-device=coralnpu \
// RUN:   --compile-to=executable-configurations \
// RUN:   --coralnpu-stack-size-kb=16 --coralnpu-dtcm-reserve-kb=12 \
// RUN:   --coralnpu-target-cpu-features=+m,+f,+zvl128b,+zve32f %s \
// RUN:   2>&1 | FileCheck %s --check-prefix=SMALL_NOZVT \
// RUN:     --implicit-check-not=warning:

// CoralNPUTileSizeSelectionDTCM runs after LLVMCPUSelectLoweringStrategyPass.
// It refines the vector tile sizes (Zvt or RVV), sets the DTCM tile sizes
// (cache_parallel, cache_reduction), and sets the distribution tile sizes to
// cache_parallel.
// For an RVV contraction, the DTCM pass also sets promote_operands, so that
// CoralNPUPromoteOperands copies the tiles to DTCM (promote_operands.mlir).
//
// The DTCM tile budget is the DTCM size minus the stack (4 KB) and the reserve
// (4 KB), rounded down to 4 KB: 24 KB for the default 32 KB DTCM (CHECK and
// DOUBLE, which counts 2 buffers per input operand), 4 KB for a 12 KB DTCM
// (SMALL) or a 16 KB stack and 12 KB reserve (SMALL_NOZVT), and 1016 KB for a
// 1 MB DTCM (HIGHMEM). The footprint of a tile is the sum of its DTCM buffers,
// each rounded up to 64 B.
// On a target with Zvt (the default; NOZVT and SMALL_NOZVT have no Zvt), a
// contraction or a convolution that the Zvt asm can take is a Zvt root: it gets
// Zvt vector tiles and no promotion. A Zvt root and a convolution stream their
// reduction loops: the DTCM tile does not split them, and the input slices hold
// one vector step of them. CoralNPUMatrixCodegen can also lower the vector
// tiles of an RVV contraction with Zvt asm. The footprint then also counts the
// scratch tiles to which the asm drains the accumulators of a result that a
// fused op reads.

// f32 matmul: Zvt streams K, so the 32x32 Zvt vector tile fits. The output
// staging buffer limits the DTCM tile: C 32x128 f32 (16 KB) + A 32x1 + B 1x128
// f32. With 4 KB, the Zvt tile is 16x32: C 16x32 f32 (2 KB). Without Zvt and
// with 4 KB (SMALL_NOZVT), the DTCM pass splits K: A 4x32 + B 32x16 + C 4x16
// f32 = 2816 B (with K = 64, B alone needs 4 KB).
// CHECK-LABEL: func.func @matmul_f32_dispatch_0
// CHECK:         linalg.matmul {lowering_config = #iree_cpu.lowering_config<cache_parallel = [32, 128, 0], cache_reduction = [0, 0, 0], distribution = [32, 128, 0], vector_common_parallel = [32, 32, 0], vector_reduction = [0, 0, 1]>}
// SMALL-LABEL: func.func @matmul_f32_dispatch_0
// SMALL:         linalg.matmul {lowering_config = #iree_cpu.lowering_config<cache_parallel = [16, 32, 0], cache_reduction = [0, 0, 0], distribution = [16, 32, 0], vector_common_parallel = [16, 32, 0], vector_reduction = [0, 0, 1]>}
// SMALL_NOZVT-LABEL: func.func @matmul_f32_dispatch_0
// SMALL_NOZVT:         linalg.matmul {lowering_config = #iree_cpu.lowering_config<cache_parallel = [4, 16, 0], cache_reduction = [0, 0, 32], distribution = [4, 16, 0], promote_operands = [0, 1, 2], vector_common_parallel = [4, 16, 0], vector_reduction = [0, 0, 1]>}
func.func @matmul_f32(%a: tensor<256x256xf32>, %b: tensor<256x256xf32>) -> tensor<256x256xf32> {
  %cst = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<256x256xf32>
  %f = linalg.fill ins(%cst : f32) outs(%e : tensor<256x256xf32>) -> tensor<256x256xf32>
  %r = linalg.matmul ins(%a, %b : tensor<256x256xf32>, tensor<256x256xf32>) outs(%f : tensor<256x256xf32>) -> tensor<256x256xf32>
  return %r : tensor<256x256xf32>
}

// Zvt matmul: the DTCM pass sets the 32x32 Zvt vector tile, enables loop
// peeling, and erases the lowering config of the fill. With 24 KB, the whole op
// fits: C 64x64 i32 (16 KB) + A 64x1 + B 1x64 i8. With 4 KB, the Zvt tile is
// 16x32: C 16x32 i32 (2 KB). Without Zvt and with 4 KB (SMALL_NOZVT), the op is
// on RVV with the strategy vector sizes, and the DTCM pass splits K.
// CHECK-LABEL: func.func @matmul_i8_dispatch_0
// CHECK-SAME:    translation_info = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<DoubleTilingExpert>, {enable_loop_peeling}>
// CHECK:         linalg.fill ins(
// CHECK:         linalg.matmul {lowering_config = #iree_cpu.lowering_config<cache_parallel = [0, 0, 0], cache_reduction = [0, 0, 0], distribution = [0, 0, 0], vector_common_parallel = [32, 32, 0], vector_reduction = [0, 0, 1]>}
// SMALL-LABEL: func.func @matmul_i8_dispatch_0
// SMALL:         linalg.matmul {lowering_config = #iree_cpu.lowering_config<cache_parallel = [16, 32, 0], cache_reduction = [0, 0, 0], distribution = [16, 32, 0], vector_common_parallel = [16, 32, 0], vector_reduction = [0, 0, 1]>}
// SMALL_NOZVT-LABEL: func.func @matmul_i8_dispatch_0
// SMALL_NOZVT:         linalg.matmul {lowering_config = #iree_cpu.lowering_config<cache_parallel = [2, 32, 0], cache_reduction = [0, 0, 64], distribution = [2, 32, 0], promote_operands = [0, 1, 2], vector_common_parallel = [2, 32, 0], vector_reduction = [0, 0, 1]>}
func.func @matmul_i8(%a: tensor<64x128xi8>, %b: tensor<128x64xi8>) -> tensor<64x64xi32> {
  %c0 = arith.constant 0 : i32
  %e = tensor.empty() : tensor<64x64xi32>
  %f = linalg.fill ins(%c0 : i32) outs(%e : tensor<64x64xi32>) -> tensor<64x64xi32>
  %r = linalg.matmul ins(%a, %b : tensor<64x128xi8>, tensor<128x64xi8>) outs(%f : tensor<64x64xi32>) -> tensor<64x64xi32>
  return %r : tensor<64x64xi32>
}

// Convolution (generalized): the halo of the input is part of the footprint,
// and the convolution streams its reduction loops (one KH step, and KW = 7).
// With 24 KB, the DTCM pass takes 11 output rows (121 = 11 x 11): input 11x256
// f32 (11264 B) + filter 1x7 f32 (64 B) + output 11x250 f32 (11008 B) =
// 22336 B. With 4 KB, it takes 1 output row.
// CHECK-LABEL: func.func @conv_2d_dispatch_0
// CHECK:         linalg.generic
// CHECK-SAME:      lowering_config = #iree_cpu.lowering_config<cache_parallel = [11, 0, 0, 0], cache_reduction = [0, 0, 0, 0], distribution = [11, 0, 0, 0], vector_common_parallel = [1, 8, 0, 0], vector_reduction = [0, 0, 1, 8]>
// SMALL-LABEL: func.func @conv_2d_dispatch_0
// SMALL:         linalg.generic
// SMALL-SAME:      lowering_config = #iree_cpu.lowering_config<cache_parallel = [1, 0, 0, 0], cache_reduction = [0, 0, 0, 0], distribution = [1, 0, 0, 0], vector_common_parallel = [1, 8, 0, 0], vector_reduction = [0, 0, 1, 8]>
func.func @conv_2d(%0: tensor<125x256xf32>, %1: tensor<5x7xf32>) -> tensor<121x250xf32> {
  %cst = arith.constant 0.0 : f32
  %2 = tensor.empty() : tensor<121x250xf32>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<121x250xf32>) -> tensor<121x250xf32>
  %4 = linalg.conv_2d ins(%0, %1 : tensor<125x256xf32>, tensor<5x7xf32>) outs(%3 : tensor<121x250xf32>) -> tensor<121x250xf32>
  return %4 : tensor<121x250xf32>
}

// Narrow producer: A is i8 in memory, and a fused sitofp makes it f32. The Zvt
// asm cannot load A (isZvtReadableInputs), so the op stays on RVV, with
// promotion. The input slice of A counts as i8: untiled, A 64x128 i8 (8 KB) + B
// 128x8 f32 (4 KB) + C 64x8 f32 (2 KB) = 14 KB fits in 24 KB. As f32, A alone
// would need 32 KB. With 2 buffers per input (DOUBLE), 2 * (8 KB + 4 KB) +
// 2 KB = 26 KB exceeds 24 KB, so M shrinks to 32 (17 KB).
// CHECK-LABEL: func.func @matmul_narrow_lhs_dispatch_0
// CHECK:         linalg.matmul {lowering_config = #iree_cpu.lowering_config<cache_parallel = [0, 0, 0], cache_reduction = [0, 0, 0], distribution = [0, 0, 0], promote_operands = [0, 1, 2], vector_common_parallel = [4, 16, 0], vector_reduction = [0, 0, 1]>}
// DOUBLE-LABEL: func.func @matmul_narrow_lhs_dispatch_0
// DOUBLE:        linalg.matmul {lowering_config = #iree_cpu.lowering_config<cache_parallel = [32, 0, 0], cache_reduction = [0, 0, 0], distribution = [32, 0, 0], promote_operands = [0, 1, 2], vector_common_parallel = [4, 16, 0], vector_reduction = [0, 0, 1]>}
func.func @matmul_narrow_lhs(%a: tensor<64x128xi8>, %b: tensor<128x8xf32>) -> tensor<64x8xf32> {
  %cst = arith.constant 0.0 : f32
  %ea = tensor.empty() : tensor<64x128xf32>
  %af = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
                        iterator_types = ["parallel", "parallel"]}
      ins(%a : tensor<64x128xi8>) outs(%ea : tensor<64x128xf32>) {
  ^bb0(%in: i8, %out: f32):
    %v = arith.sitofp %in : i8 to f32
    linalg.yield %v : f32
  } -> tensor<64x128xf32>
  %e = tensor.empty() : tensor<64x8xf32>
  %f = linalg.fill ins(%cst : f32) outs(%e : tensor<64x8xf32>) -> tensor<64x8xf32>
  %r = linalg.matmul ins(%af, %b : tensor<64x128xf32>, tensor<128x8xf32>) outs(%f : tensor<64x8xf32>) -> tensor<64x8xf32>
  return %r : tensor<64x8xf32>
}

// Bias, output staging, Zvt scratch and split-K accumulator: i8 x i8 -> i32,
// then a fused consumer adds an i32 bias and truncates to i8. The op is a Zvt
// root, and the Zvt asm drains the 32x32 vector tiles to a 32x32 i32 scratch
// tile (4 KB) for the consumer. With 24 KB, the whole op fits: output 64x64 i8
// (4 KB) + scratch (4 KB) + bias 64 i32 (256 B) + A 64x1 + B 1x64 i8. With 4 KB,
// the 32x32 Zvt tile does not fit, so the Zvt tile is 16x32 (a 16x32 i32
// scratch tile, 2 KB), and the DTCM tile has 16 rows: output 16x64 i8 (1 KB) +
// scratch (2 KB) + bias (256 B) + A + B = 3456 B.
// Without Zvt (NOZVT), there is no scratch tile: A 32x224 i8 (7 KB) + B (14 KB)
// + bias (256 B) + output 32x64 i8 (2 KB) = 23.25 KB.
// Without Zvt and with 4 KB (SMALL_NOZVT), the tile splits K, so the i32
// accumulator (2x32, 256 B) needs a buffer too: with K = 112, A 2x112 i8
// (256 B) + B 112x32 i8 (3584 B) + bias 32 i32 (128 B) + output 2x32 i8 (64 B)
// + accumulator (256 B) = 4288 B, which does not fit, so K = 56.
// CHECK-LABEL: func.func @matmul_bias_trunc_dispatch_0
// CHECK:         linalg.matmul {lowering_config = #iree_cpu.lowering_config<cache_parallel = [0, 0, 0], cache_reduction = [0, 0, 0], distribution = [0, 0, 0], vector_common_parallel = [32, 32, 0], vector_reduction = [0, 0, 1]>}
// SMALL-LABEL: func.func @matmul_bias_trunc_dispatch_0
// SMALL:         linalg.matmul {lowering_config = #iree_cpu.lowering_config<cache_parallel = [16, 0, 0], cache_reduction = [0, 0, 0], distribution = [16, 0, 0], vector_common_parallel = [16, 32, 0], vector_reduction = [0, 0, 1]>}
// NOZVT-LABEL: func.func @matmul_bias_trunc_dispatch_0
// NOZVT:         linalg.matmul {lowering_config = #iree_cpu.lowering_config<cache_parallel = [32, 0, 0], cache_reduction = [0, 0, 0], distribution = [32, 0, 0], promote_operands = [0, 1, 2], vector_common_parallel = [2, 32, 0], vector_reduction = [0, 0, 1]>}
// SMALL_NOZVT-LABEL: func.func @matmul_bias_trunc_dispatch_0
// SMALL_NOZVT:         linalg.matmul {lowering_config = #iree_cpu.lowering_config<cache_parallel = [2, 32, 0], cache_reduction = [0, 0, 56], distribution = [2, 32, 0], promote_operands = [0, 1, 2], vector_common_parallel = [2, 32, 0], vector_reduction = [0, 0, 1]>}
func.func @matmul_bias_trunc(%a: tensor<64x224xi8>, %b: tensor<224x64xi8>, %bias: tensor<64xi32>) -> tensor<64x64xi8> {
  %c0 = arith.constant 0 : i32
  %e = tensor.empty() : tensor<64x64xi32>
  %f = linalg.fill ins(%c0 : i32) outs(%e : tensor<64x64xi32>) -> tensor<64x64xi32>
  %r = linalg.matmul ins(%a, %b : tensor<64x224xi8>, tensor<224x64xi8>) outs(%f : tensor<64x64xi32>) -> tensor<64x64xi32>
  %eo = tensor.empty() : tensor<64x64xi8>
  %o = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>],
                       iterator_types = ["parallel", "parallel"]}
      ins(%r, %bias : tensor<64x64xi32>, tensor<64xi32>) outs(%eo : tensor<64x64xi8>) {
  ^bb0(%in: i32, %bi: i32, %out: i8):
    %s = arith.addi %in, %bi : i32
    %t = arith.trunci %s : i32 to i8
    linalg.yield %t : i8
  } -> tensor<64x64xi8>
  return %o : tensor<64x64xi8>
}

// Zvt scratch of an RVV root: with 2 rows, the op is not a Zvt root (Zvt needs
// 3 rows or more), so it keeps the strategy vector tile (2x32) and gets
// promotion. The Zvt asm takes the 2x32 vector tiles in the K loop, and drains
// them to a 16x32 i32 scratch tile (2 KB) for the consumer. With 24 KB, the
// full N needs A 2x352 i8 (704 B) + B 352x64 i8 (22 KB) + bias 64 i32 (256 B)
// + output 2x64 i8 (128 B) + scratch (2 KB) = 25664 B, so N is halved: A +
// B 352x32 i8 (11 KB) + bias (128 B) + output (64 B) + scratch = 14208 B.
// Without Zvt (NOZVT), there is no scratch tile, and the full N (23616 B) fits.
// CHECK-LABEL: func.func @matmul_bias_trunc_m2_dispatch_0
// CHECK:         linalg.matmul {lowering_config = #iree_cpu.lowering_config<cache_parallel = [0, 32, 0], cache_reduction = [0, 0, 0], distribution = [0, 32, 0], promote_operands = [0, 1, 2], vector_common_parallel = [2, 32, 0], vector_reduction = [0, 0, 1]>}
// NOZVT-LABEL: func.func @matmul_bias_trunc_m2_dispatch_0
// NOZVT:         linalg.matmul {lowering_config = #iree_cpu.lowering_config<cache_parallel = [0, 0, 0], cache_reduction = [0, 0, 0], distribution = [0, 0, 0], promote_operands = [0, 1, 2], vector_common_parallel = [2, 32, 0], vector_reduction = [0, 0, 1]>}
func.func @matmul_bias_trunc_m2(%a: tensor<2x352xi8>, %b: tensor<352x64xi8>, %bias: tensor<64xi32>) -> tensor<2x64xi8> {
  %c0 = arith.constant 0 : i32
  %e = tensor.empty() : tensor<2x64xi32>
  %f = linalg.fill ins(%c0 : i32) outs(%e : tensor<2x64xi32>) -> tensor<2x64xi32>
  %r = linalg.matmul ins(%a, %b : tensor<2x352xi8>, tensor<352x64xi8>) outs(%f : tensor<2x64xi32>) -> tensor<2x64xi32>
  %eo = tensor.empty() : tensor<2x64xi8>
  %o = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>],
                       iterator_types = ["parallel", "parallel"]}
      ins(%r, %bias : tensor<2x64xi32>, tensor<64xi32>) outs(%eo : tensor<2x64xi8>) {
  ^bb0(%in: i32, %bi: i32, %out: i8):
    %s = arith.addi %in, %bi : i32
    %t = arith.trunci %s : i32 to i8
    linalg.yield %t : i8
  } -> tensor<2x64xi8>
  return %o : tensor<2x64xi8>
}

// Vector clamp: a user config sets a 16x64 vector tile (i32 is not a Zvt
// type). With 24 KB, the DTCM tile is the vector tile: A 16x32 + B 32x64 + C
// 16x64 i32 = 14 KB. With 4 KB, C 16x64 i32 alone needs 4 KB, so the search
// relaxes the alignments to 1, and the DTCM pass clamps the vector tile to the
// 8x64x1 DTCM tile.
// CHECK-LABEL: func.func @matmul_clamp_dispatch_0
// CHECK:         linalg.matmul {lowering_config = #iree_cpu.lowering_config<cache_parallel = [16, 64, 0], cache_reduction = [0, 0, 0], distribution = [16, 64, 0], promote_operands = [0, 1, 2], vector_common_parallel = [16, 64, 0], vector_reduction = [0, 0, 1]>}
// SMALL-LABEL: func.func @matmul_clamp_dispatch_0
// SMALL:         linalg.matmul {lowering_config = #iree_cpu.lowering_config<cache_parallel = [8, 64, 0], cache_reduction = [0, 0, 1], distribution = [8, 64, 0], promote_operands = [0, 1, 2], vector_common_parallel = [8, 64, 0], vector_reduction = [0, 0, 1]>}
#config = #iree_cpu.lowering_config<distribution = [32, 128, 0],
    vector_common_parallel = [16, 64, 0], vector_reduction = [0, 0, 1]>
#translation = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<DoubleTilingExpert>>
#compilation = #iree_codegen.compilation_info<lowering_config = #config,
    translation_info = #translation>
func.func @matmul_clamp(%a: tensor<32x32xi32>, %b: tensor<32x128xi32>) -> tensor<32x128xi32> {
  %c0 = arith.constant 0 : i32
  %e = tensor.empty() : tensor<32x128xi32>
  %f = linalg.fill ins(%c0 : i32) outs(%e : tensor<32x128xi32>) -> tensor<32x128xi32>
  %r = linalg.matmul {compilation_info = #compilation}
      ins(%a, %b : tensor<32x32xi32>, tensor<32x128xi32>) outs(%f : tensor<32x128xi32>) -> tensor<32x128xi32>
  return %r : tensor<32x128xi32>
}

// 1 MB DTCM: untiled. The op is a Zvt root, which streams K: C 288x288 f32
// (324 KB) + A 288x1 + B 1x288 f32 fits in the 1016 KB budget (with 24 KB, the
// DTCM tile is 32x96).
// HIGHMEM-LABEL: func.func @matmul_highmem_dispatch_0
// HIGHMEM:         linalg.matmul {lowering_config = #iree_cpu.lowering_config<cache_parallel = [0, 0, 0], cache_reduction = [0, 0, 0], distribution = [0, 0, 0], {{.*}}>}
func.func @matmul_highmem(%a: tensor<288x288xf32>, %b: tensor<288x288xf32>) -> tensor<288x288xf32> {
  %cst = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<288x288xf32>
  %f = linalg.fill ins(%cst : f32) outs(%e : tensor<288x288xf32>) -> tensor<288x288xf32>
  %r = linalg.matmul ins(%a, %b : tensor<288x288xf32>, tensor<288x288xf32>) outs(%f : tensor<288x288xf32>) -> tensor<288x288xf32>
  return %r : tensor<288x288xf32>
}

// A 3x3 convolution cannot peel N, so N = 24 takes two 12-column Zvt tiles
// rather than a 16-column one and an 8-column remainder. The Zvt asm takes a K
// of 1, so all reduction loops get the vector size 1 (the strategy gives IC 8).
// CHECK-LABEL: func.func @conv_3x3_n24
// CHECK: linalg.generic
// CHECK-SAME: vector_common_parallel = [1, 1, 0, 12, 0, 0, 0], vector_reduction = [0, 0, 0, 0, 1, 1, 1]
func.func @conv_3x3_n24(%in: tensor<1x34x34x16xi8>, %w: tensor<3x3x16x24xi8>) -> tensor<1x32x32x24xi32> {
  %c0 = arith.constant 0 : i32
  %e = tensor.empty() : tensor<1x32x32x24xi32>
  %f = linalg.fill ins(%c0 : i32) outs(%e : tensor<1x32x32x24xi32>) -> tensor<1x32x32x24xi32>
  %r = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
       ins(%in, %w : tensor<1x34x34x16xi8>, tensor<3x3x16x24xi8>) outs(%f : tensor<1x32x32x24xi32>) -> tensor<1x32x32x24xi32>
  return %r : tensor<1x32x32x24xi32>
}

// N = 20 splits into neither 12- nor 8-column Zvt tiles, so the convolution
// uses RVV with ConvTileAndDecomposeExpert (`ow = 4`, `oc = 20`, so `oc` is
// untiled at the vector level).
// CHECK-LABEL: func.func @conv_3x3_n20
// CHECK-SAME: pipeline = #iree_cpu.pipeline<ConvTileAndDecomposeExpert>
// CHECK: linalg.generic
// CHECK-SAME: vector_common_parallel = [0, 1, 4, 0, 0, 0, 0], vector_reduction = [0, 0, 0, 0, 1, 1, 1]
func.func @conv_3x3_n20(%in: tensor<1x34x34x16xi8>, %w: tensor<3x3x16x20xi8>) -> tensor<1x32x32x20xi32> {
  %c0 = arith.constant 0 : i32
  %e = tensor.empty() : tensor<1x32x32x20xi32>
  %f = linalg.fill ins(%c0 : i32) outs(%e : tensor<1x32x32x20xi32>) -> tensor<1x32x32x20xi32>
  %r = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
       ins(%in, %w : tensor<1x34x34x16xi8>, tensor<3x3x16x20xi8>) outs(%f : tensor<1x32x32x20xi32>) -> tensor<1x32x32x20xi32>
  return %r : tensor<1x32x32x20xi32>
}

// A 2D NHWC convolution with F = 1 has its unit batch and output channel
// dimensions folded away into a 5-loop generic (`outputChannel` is empty), so
// RVV falls back to the non-channel-last 2D conv rule (`[1, 16, 0, 0, 0]`).
// CHECK-LABEL: func.func @conv_3x3_f1
// CHECK-SAME: pipeline = #iree_cpu.pipeline<DoubleTilingExpert>
// CHECK: linalg.generic
// CHECK-SAME: vector_common_parallel = [1, 16, 0, 0, 0], vector_reduction = [0, 0, 1, 1, 1]
func.func @conv_3x3_f1(%in: tensor<1x34x34x16xf32>, %w: tensor<3x3x16x1xf32>) -> tensor<1x32x32x1xf32> {
  %c0 = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<1x32x32x1xf32>
  %f = linalg.fill ins(%c0 : f32) outs(%e : tensor<1x32x32x1xf32>) -> tensor<1x32x32x1xf32>
  %r = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
       ins(%in, %w : tensor<1x34x34x16xf32>, tensor<3x3x16x1xf32>) outs(%f : tensor<1x32x32x1xf32>) -> tensor<1x32x32x1xf32>
  return %r : tensor<1x32x32x1xf32>
}

// Widening 2D NHWC depthwise conv (c = 120, i8 -> i32): RVV snaps c from 32
// to 30 (cRegs = 8), sets ow = 16 / 8 = 2, and unrolls kw = 3 (to 0). The
// DTCM pass pre-tiles c to 30 before shrinking oh to 4 so the tile keeps a
// taller spatial frame with fewer duplicated halo rows.
// CHECK-LABEL: func.func @depthwise_3x3_c120
// CHECK-SAME: pipeline = #iree_cpu.pipeline<ConvTileAndDecomposeExpert>
// CHECK: linalg.generic
// CHECK-SAME: cache_parallel = [0, 4, 0, 30, 0, 0], cache_reduction = [0, 0, 0, 0, 0, 0], distribution = [0, 4, 0, 30, 0, 0]
// CHECK-SAME: vector_common_parallel = [0, 1, 2, 30, 0, 0], vector_reduction = [0, 0, 0, 0, 1, 0]
func.func @depthwise_3x3_c120(%in: tensor<1x34x34x120xi8>, %w: tensor<3x3x120xi8>) -> tensor<1x32x32x120xi32> {
  %c0 = arith.constant 0 : i32
  %e = tensor.empty() : tensor<1x32x32x120xi32>
  %f = linalg.fill ins(%c0 : i32) outs(%e : tensor<1x32x32x120xi32>) -> tensor<1x32x32x120xi32>
  %r = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
       ins(%in, %w : tensor<1x34x34x120xi8>, tensor<3x3x120xi8>) outs(%f : tensor<1x32x32x120xi32>) -> tensor<1x32x32x120xi32>
  return %r : tensor<1x32x32x120xi32>
}

// 1x4 and 4x1 Zvt strips: N = 128 takes 64-column tiles, M = 128 64-row ones.
// CHECK-LABEL: func.func @matmul_16x128
// CHECK: linalg.matmul
// CHECK-SAME: vector_common_parallel = [0, 64, 0]
func.func @matmul_16x128(%a: tensor<16x128xf32>, %b: tensor<128x128xf32>) -> tensor<16x128xf32> {
  %c0 = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<16x128xf32>
  %f = linalg.fill ins(%c0 : f32) outs(%e : tensor<16x128xf32>) -> tensor<16x128xf32>
  %r = linalg.matmul ins(%a, %b : tensor<16x128xf32>, tensor<128x128xf32>) outs(%f : tensor<16x128xf32>) -> tensor<16x128xf32>
  return %r : tensor<16x128xf32>
}

// CHECK-LABEL: func.func @matmul_128x16
// CHECK: linalg.matmul
// CHECK-SAME: vector_common_parallel = [64, 0, 0]
func.func @matmul_128x16(%a: tensor<128x128xf32>, %b: tensor<128x16xf32>) -> tensor<128x16xf32> {
  %c0 = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<128x16xf32>
  %f = linalg.fill ins(%c0 : f32) outs(%e : tensor<128x16xf32>) -> tensor<128x16xf32>
  %r = linalg.matmul ins(%a, %b : tensor<128x128xf32>, tensor<128x16xf32>) outs(%f : tensor<128x16xf32>) -> tensor<128x16xf32>
  return %r : tensor<128x16xf32>
}

// With a partial N, M = 196 needs whole M tiles: 14 rows (196 = 14 x 14).
// CHECK-LABEL: func.func @matmul_m196_n8
// CHECK: linalg.matmul
// CHECK-SAME: vector_common_parallel = [14, 0, 0]
func.func @matmul_m196_n8(%a: tensor<196x64xi8>, %b: tensor<64x8xi8>) -> tensor<196x8xi32> {
  %c0 = arith.constant 0 : i32
  %e = tensor.empty() : tensor<196x8xi32>
  %f = linalg.fill ins(%c0 : i32) outs(%e : tensor<196x8xi32>) -> tensor<196x8xi32>
  %r = linalg.matmul ins(%a, %b : tensor<196x64xi8>, tensor<64x8xi8>) outs(%f : tensor<196x8xi32>) -> tensor<196x8xi32>
  return %r : tensor<196x8xi32>
}

// N = 1040 takes 64-column tiles and a peeled 16-column remainder. The 16 x N
// accumulator does not fit the DTCM, so N shrinks by whole 64-column tiles
// (256) rather than to an exact divisor (260) with a 4-column remainder.
// CHECK-LABEL: func.func @matmul_n1040
// CHECK-SAME: enable_loop_peeling
// CHECK: linalg.matmul
// CHECK-SAME: distribution = [0, 256, 0]
// CHECK-SAME: vector_common_parallel = [0, 64, 0]
func.func @matmul_n1040(%a: tensor<16x64xi8>, %b: tensor<64x1040xi8>) -> tensor<16x1040xi32> {
  %c0 = arith.constant 0 : i32
  %e = tensor.empty() : tensor<16x1040xi32>
  %f = linalg.fill ins(%c0 : i32) outs(%e : tensor<16x1040xi32>) -> tensor<16x1040xi32>
  %r = linalg.matmul ins(%a, %b : tensor<16x64xi8>, tensor<64x1040xi8>) outs(%f : tensor<16x1040xi32>) -> tensor<16x1040xi32>
  return %r : tensor<16x1040xi32>
}

// N = 36 takes 12-column tiles: 16-column ones would leave a 4-column
// remainder that the Zvt codegen cannot lower.
// CHECK-LABEL: func.func @matmul_n36
// CHECK: linalg.matmul
// CHECK-SAME: vector_common_parallel = [0, 12, 0]
func.func @matmul_n36(%a: tensor<32x64xi8>, %b: tensor<64x36xi8>) -> tensor<32x36xi32> {
  %c0 = arith.constant 0 : i32
  %e = tensor.empty() : tensor<32x36xi32>
  %f = linalg.fill ins(%c0 : i32) outs(%e : tensor<32x36xi32>) -> tensor<32x36xi32>
  %r = linalg.matmul ins(%a, %b : tensor<32x64xi8>, tensor<64x36xi8>) outs(%f : tensor<32x36xi32>) -> tensor<32x36xi32>
  return %r : tensor<32x36xi32>
}

// Swapped inputs (the first is K x N) with a partial N have no Zvt lowering,
// so the contraction keeps the RVV sizes.
// CHECK-LABEL: func.func @matmul_swapped_n12
// CHECK: linalg.generic
// CHECK-SAME: vector_common_parallel = [2, 32, 0]
func.func @matmul_swapped_n12(%a: tensor<64x12xi8>, %b: tensor<32x64xi8>) -> tensor<32x12xi32> {
  %c0 = arith.constant 0 : i32
  %e = tensor.empty() : tensor<32x12xi32>
  %f = linalg.fill ins(%c0 : i32) outs(%e : tensor<32x12xi32>) -> tensor<32x12xi32>
  %r = linalg.generic {indexing_maps = [affine_map<(m, n, k) -> (k, n)>, affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (m, n)>],
                       iterator_types = ["parallel", "parallel", "reduction"]}
       ins(%a, %b : tensor<64x12xi8>, tensor<32x64xi8>) outs(%f : tensor<32x12xi32>) {
  ^bb0(%x: i8, %y: i8, %acc: i32):
    %xe = arith.extsi %x : i8 to i32
    %ye = arith.extsi %y : i8 to i32
    %p = arith.muli %xe, %ye : i32
    %s = arith.addi %acc, %p : i32
    linalg.yield %s : i32
  } -> tensor<32x12xi32>
  return %r : tensor<32x12xi32>
}

// K = 8192 does not fit the DTCM in full, but K is streamed rather than split.
// CHECK-LABEL: func.func @matmul_k8192
// CHECK: linalg.matmul
// CHECK-SAME: cache_reduction = [0, 0, 0]
// CHECK-SAME: vector_common_parallel = [0, 32, 0], vector_reduction = [0, 0, 1]
func.func @matmul_k8192(%a: tensor<32x8192xi8>, %b: tensor<8192x64xi8>) -> tensor<32x64xi32> {
  %c0 = arith.constant 0 : i32
  %e = tensor.empty() : tensor<32x64xi32>
  %f = linalg.fill ins(%c0 : i32) outs(%e : tensor<32x64xi32>) -> tensor<32x64xi32>
  %r = linalg.matmul ins(%a, %b : tensor<32x8192xi8>, tensor<8192x64xi8>) outs(%f : tensor<32x64xi32>) -> tensor<32x64xi32>
  return %r : tensor<32x64xi32>
}

// Zvt-to-RVV fallback: with 24 KB, the 32x32 Zvt tile fits untiled (scale 4 KB
// + res 4 KB + output 4 KB + scratch 4 KB + A 128 B + B 128 B = 16.25 KB).
// With 2 buffers per input (DOUBLE), 32x32 needs 24.5 KB, so Zvt halves to
// 16x32. With 4 KB (SMALL), even the minimum 16x16 Zvt tile needs 4224 B, so
// setZvtVectorSizes returns false and the search falls back to RVV and splits K.
// CHECK-LABEL: func.func @matmul_zvt_fallback_dispatch_0
// CHECK:         linalg.matmul {lowering_config = #iree_cpu.lowering_config<cache_parallel = [0, 0, 0], cache_reduction = [0, 0, 0], distribution = [0, 0, 0], vector_common_parallel = [0, 0, 0], vector_reduction = [0, 0, 1]>}
// DOUBLE-LABEL: func.func @matmul_zvt_fallback_dispatch_0
// DOUBLE:        linalg.matmul {lowering_config = #iree_cpu.lowering_config<cache_parallel = [0, 0, 0], cache_reduction = [0, 0, 0], distribution = [0, 0, 0], vector_common_parallel = [16, 0, 0], vector_reduction = [0, 0, 1]>}
// SMALL-LABEL: func.func @matmul_zvt_fallback_dispatch_0
// SMALL:         linalg.matmul {lowering_config = #iree_cpu.lowering_config<cache_parallel = [4, 16, 0], cache_reduction = [0, 0, 16], distribution = [4, 16, 0], promote_operands = [0, 1, 2], vector_common_parallel = [4, 16, 0], vector_reduction = [0, 0, 1]>}
func.func @matmul_zvt_fallback(%a: tensor<32x32xf32>, %b: tensor<32x32xf32>, %scale: tensor<32x32xf32>, %res: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %cst = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<32x32xf32>
  %f = linalg.fill ins(%cst : f32) outs(%e : tensor<32x32xf32>) -> tensor<32x32xf32>
  %mm = linalg.matmul ins(%a, %b : tensor<32x32xf32>, tensor<32x32xf32>) outs(%f : tensor<32x32xf32>) -> tensor<32x32xf32>
  %o = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
                       iterator_types = ["parallel", "parallel"]}
      ins(%mm, %scale, %res : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) outs(%e : tensor<32x32xf32>) {
  ^bb0(%in: f32, %sc: f32, %re: f32, %out: f32):
    %p = arith.mulf %in, %sc : f32
    %s = arith.addf %p, %re : f32
    linalg.yield %s : f32
  } -> tensor<32x32xf32>
  return %o : tensor<32x32xf32>
}

// Decomposed softmax (max reduction -> exp+sum root reduction -> div 2D
// consumer): non-root ops share the root's [64, 256] domain and innermost
// reduction loop, so cache_reduction stays [0, 0] and RVV refines all three ops
// to vector_common_parallel = [1, 0] with reduction/inner_parallel = [0, 16].
// Input + output is 2 KB/row (128 KB untiled). With 24 KB, 8 rows (16 KB) fit.
// With 4 KB (SMALL), 2 rows (4 KB) fit. With 1 MB (HIGHMEM), it stays untiled.
// CHECK-LABEL: func.func @softmax_dispatch_0
// CHECK:         linalg.generic
// CHECK-SAME:      lowering_config = #iree_cpu.lowering_config<vector_common_parallel = [1, 0], vector_reduction = [0, 16]>
// CHECK:         linalg.generic
// CHECK-SAME:      lowering_config = #iree_cpu.lowering_config<cache_parallel = [8, 0], cache_reduction = [0, 0], distribution = [8, 0], vector_common_parallel = [1, 0], vector_reduction = [0, 16]>
// CHECK:         linalg.generic
// CHECK-SAME:      lowering_config = #iree_cpu.lowering_config<vector_common_parallel = [1, 0], vector_inner_parallel = [0, 16]>
// SMALL-LABEL: func.func @softmax_dispatch_0
// SMALL:         linalg.generic
// SMALL-SAME:      lowering_config = #iree_cpu.lowering_config<vector_common_parallel = [1, 0], vector_reduction = [0, 16]>
// SMALL:         linalg.generic
// SMALL-SAME:      lowering_config = #iree_cpu.lowering_config<cache_parallel = [2, 0], cache_reduction = [0, 0], distribution = [2, 0], vector_common_parallel = [1, 0], vector_reduction = [0, 16]>
// SMALL:         linalg.generic
// SMALL-SAME:      lowering_config = #iree_cpu.lowering_config<vector_common_parallel = [1, 0], vector_inner_parallel = [0, 16]>
// HIGHMEM-LABEL: func.func @softmax_dispatch_0
// HIGHMEM:         linalg.generic
// HIGHMEM-SAME:      lowering_config = #iree_cpu.lowering_config<vector_common_parallel = [1, 0], vector_reduction = [0, 16]>
// HIGHMEM:         linalg.generic
// HIGHMEM-SAME:      lowering_config = #iree_cpu.lowering_config<cache_parallel = [0, 0], cache_reduction = [0, 0], distribution = [0, 0], vector_common_parallel = [1, 0], vector_reduction = [0, 16]>
// HIGHMEM:         linalg.generic
// HIGHMEM-SAME:      lowering_config = #iree_cpu.lowering_config<vector_common_parallel = [1, 0], vector_inner_parallel = [0, 16]>
func.func @softmax(%x: tensor<64x256xf32>) -> tensor<64x256xf32> {
  %e = tensor.empty() : tensor<64x256xf32>
  %r = linalg.softmax dimension(1) ins(%x : tensor<64x256xf32>) outs(%e : tensor<64x256xf32>) -> tensor<64x256xf32>
  return %r : tensor<64x256xf32>
}

// Softmax fused with a 2D post-scale consumer: 2 inputs (x, w) + 1 output of
// 16x512 f32 need 6 KB/row (96 KB untiled). With 24 KB, 4 rows (24 KB) fit.
// With 4 KB (SMALL), even 1 row (6 KB) exceeds the budget and cache_reduction
// cannot split, so the pass keeps the 1-row parallel tile [1, 0] without
// emitting a warning. With 1 MB (HIGHMEM), it stays untiled.
// CHECK-LABEL: func.func @softmax_scaled_dispatch_0
// CHECK:         linalg.generic
// CHECK-SAME:      lowering_config = #iree_cpu.lowering_config<vector_common_parallel = [1, 0], vector_reduction = [0, 16]>
// CHECK:         linalg.generic
// CHECK-SAME:      lowering_config = #iree_cpu.lowering_config<cache_parallel = [4, 0], cache_reduction = [0, 0], distribution = [4, 0], vector_common_parallel = [1, 0], vector_reduction = [0, 16]>
// CHECK:         linalg.generic
// CHECK-SAME:      lowering_config = #iree_cpu.lowering_config<vector_common_parallel = [1, 0], vector_inner_parallel = [0, 16]>
// SMALL-LABEL: func.func @softmax_scaled_dispatch_0
// SMALL:         linalg.generic
// SMALL-SAME:      lowering_config = #iree_cpu.lowering_config<vector_common_parallel = [1, 0], vector_reduction = [0, 16]>
// SMALL:         linalg.generic
// SMALL-SAME:      lowering_config = #iree_cpu.lowering_config<cache_parallel = [1, 0], cache_reduction = [0, 0], distribution = [1, 0], vector_common_parallel = [1, 0], vector_reduction = [0, 16]>
// SMALL:         linalg.generic
// SMALL-SAME:      lowering_config = #iree_cpu.lowering_config<vector_common_parallel = [1, 0], vector_inner_parallel = [0, 16]>
// HIGHMEM-LABEL: func.func @softmax_scaled_dispatch_0
// HIGHMEM:         linalg.generic
// HIGHMEM-SAME:      lowering_config = #iree_cpu.lowering_config<vector_common_parallel = [1, 0], vector_reduction = [0, 16]>
// HIGHMEM:         linalg.generic
// HIGHMEM-SAME:      lowering_config = #iree_cpu.lowering_config<cache_parallel = [0, 0], cache_reduction = [0, 0], distribution = [0, 0], vector_common_parallel = [1, 0], vector_reduction = [0, 16]>
// HIGHMEM:         linalg.generic
// HIGHMEM-SAME:      lowering_config = #iree_cpu.lowering_config<vector_common_parallel = [1, 0], vector_inner_parallel = [0, 16]>
func.func @softmax_scaled(%x: tensor<16x512xf32>, %w: tensor<16x512xf32>) -> tensor<16x512xf32> {
  %e = tensor.empty() : tensor<16x512xf32>
  %s = linalg.softmax dimension(1) ins(%x : tensor<16x512xf32>) outs(%e : tensor<16x512xf32>) -> tensor<16x512xf32>
  %r = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
                       iterator_types = ["parallel", "parallel"]}
      ins(%s, %w : tensor<16x512xf32>, tensor<16x512xf32>) outs(%e : tensor<16x512xf32>) {
  ^bb0(%in0: f32, %in1: f32, %out: f32):
    %p = arith.mulf %in0, %in1 : f32
    linalg.yield %p : f32
  } -> tensor<16x512xf32>
  return %r : tensor<16x512xf32>
}

// Unaligned softmax row count (196x256 f32, 2 KB/row, 392 KB untiled):
// RVV refines vector_common_parallel to [1, 0] and vector_reduction to [0, 16],
// so DTCM shrinks along exact divisors of 196: 7 rows (14 KB) for 24 KB DTCM,
// 2 rows (4 KB) for 4 KB DTCM (SMALL), and untiled [0, 0] for 1 MB (HIGHMEM).
// CHECK-LABEL: func.func @softmax_unaligned_dispatch_0
// CHECK:         linalg.generic
// CHECK-SAME:      lowering_config = #iree_cpu.lowering_config<vector_common_parallel = [1, 0], vector_reduction = [0, 16]>
// CHECK:         linalg.generic
// CHECK-SAME:      lowering_config = #iree_cpu.lowering_config<cache_parallel = [7, 0], cache_reduction = [0, 0], distribution = [7, 0], vector_common_parallel = [1, 0], vector_reduction = [0, 16]>
// CHECK:         linalg.generic
// CHECK-SAME:      lowering_config = #iree_cpu.lowering_config<vector_common_parallel = [1, 0], vector_inner_parallel = [0, 16]>
// SMALL-LABEL: func.func @softmax_unaligned_dispatch_0
// SMALL:         linalg.generic
// SMALL-SAME:      lowering_config = #iree_cpu.lowering_config<vector_common_parallel = [1, 0], vector_reduction = [0, 16]>
// SMALL:         linalg.generic
// SMALL-SAME:      lowering_config = #iree_cpu.lowering_config<cache_parallel = [2, 0], cache_reduction = [0, 0], distribution = [2, 0], vector_common_parallel = [1, 0], vector_reduction = [0, 16]>
// SMALL:         linalg.generic
// SMALL-SAME:      lowering_config = #iree_cpu.lowering_config<vector_common_parallel = [1, 0], vector_inner_parallel = [0, 16]>
// HIGHMEM-LABEL: func.func @softmax_unaligned_dispatch_0
// HIGHMEM:         linalg.generic
// HIGHMEM-SAME:      lowering_config = #iree_cpu.lowering_config<vector_common_parallel = [1, 0], vector_reduction = [0, 16]>
// HIGHMEM:         linalg.generic
// HIGHMEM-SAME:      lowering_config = #iree_cpu.lowering_config<cache_parallel = [0, 0], cache_reduction = [0, 0], distribution = [0, 0], vector_common_parallel = [1, 0], vector_reduction = [0, 16]>
// HIGHMEM:         linalg.generic
// HIGHMEM-SAME:      lowering_config = #iree_cpu.lowering_config<vector_common_parallel = [1, 0], vector_inner_parallel = [0, 16]>
func.func @softmax_unaligned(%x: tensor<196x256xf32>) -> tensor<196x256xf32> {
  %e = tensor.empty() : tensor<196x256xf32>
  %r = linalg.softmax dimension(1) ins(%x : tensor<196x256xf32>) outs(%e : tensor<196x256xf32>) -> tensor<196x256xf32>
  return %r : tensor<196x256xf32>
}

// A vector-matrix product has no reuse across M, so it streams K without DTCM
// operand promotion. With 24 KB, the full N = 1024 fits: x 1 f32 (64 B) +
// w 1x1024 f32 (4 KB) + output 1024 f32 (4 KB) = 8256 B. With 4 KB (SMALL),
// N shrinks to 256 (2112 B) while K stays unsplit and unpromoted.
// CHECK-LABEL: func.func @vecmat_f32_dispatch_0
// CHECK:         linalg.vecmat
// CHECK-SAME:      cache_parallel = [0, 0], cache_reduction = [0, 0], distribution = [0, 0]
// CHECK-NOT:       promote_operands
// CHECK-SAME:      vector_common_parallel = [32, 0], vector_reduction = [0, 1]
// SMALL-LABEL: func.func @vecmat_f32_dispatch_0
// SMALL:         linalg.vecmat
// SMALL-SAME:      cache_parallel = [256, 0], cache_reduction = [0, 0], distribution = [256, 0]
// SMALL-NOT:       promote_operands
// SMALL-SAME:      vector_common_parallel = [32, 0], vector_reduction = [0, 1]
func.func @vecmat_f32(%x: tensor<640xf32>, %w: tensor<640x1024xf32>) -> tensor<1024xf32> {
  %cst = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<1024xf32>
  %f = linalg.fill ins(%cst : f32) outs(%e : tensor<1024xf32>) -> tensor<1024xf32>
  %r = linalg.vecmat ins(%x, %w : tensor<640xf32>, tensor<640x1024xf32>) outs(%f : tensor<1024xf32>) -> tensor<1024xf32>
  return %r : tensor<1024xf32>
}

// Widening integer vecmat (512xi8 * 512x1000xi8 -> 1000xi32): 1000 % 64 = 40 >
// 32, so RVV picks one LMUL=8 accumulator group (32) and peels the 8-element
// remainder (`enable_loop_peeling`).
// CHECK-LABEL: func.func @vecmat_i8_i32_1000_dispatch_0
// CHECK-SAME:    pipeline = #iree_cpu.pipeline<DoubleTilingExpert>, {enable_loop_peeling}
// CHECK:         linalg.generic
// CHECK-SAME:      vector_common_parallel = [32, 0], vector_reduction = [0, 1]
func.func @vecmat_i8_i32_1000(%x: tensor<512xi8>, %w: tensor<512x1000xi8>) -> tensor<1000xi32> {
  %c0 = arith.constant 0 : i32
  %e = tensor.empty() : tensor<1000xi32>
  %f = linalg.fill ins(%c0 : i32) outs(%e : tensor<1000xi32>) -> tensor<1000xi32>
  %r = linalg.generic {indexing_maps = [affine_map<(n, k) -> (k)>, affine_map<(n, k) -> (k, n)>, affine_map<(n, k) -> (n)>],
                       iterator_types = ["parallel", "reduction"]}
       ins(%x, %w : tensor<512xi8>, tensor<512x1000xi8>) outs(%f : tensor<1000xi32>) {
  ^bb0(%a: i8, %b: i8, %o: i32):
    %a32 = arith.extsi %a : i8 to i32
    %b32 = arith.extsi %b : i8 to i32
    %m = arith.muli %a32, %b32 : i32
    %s = arith.addi %o, %m : i32
    linalg.yield %s : i32
  } -> tensor<1000xi32>
  return %r : tensor<1000xi32>
}

// 2D transpose (64x64xf32): RVV tiles both parallel dimensions to 8x8 for
// in-register transpose shuffles.
// CHECK-LABEL: func.func @transpose_2d_f32_dispatch_0
// CHECK-SAME:    pipeline = #iree_cpu.pipeline<DoubleTilingExpert>
// CHECK:         linalg.generic
// CHECK-SAME:      vector_common_parallel = [8, 8]
func.func @transpose_2d_f32(%x: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %e = tensor.empty() : tensor<64x64xf32>
  %r = linalg.transpose ins(%x : tensor<64x64xf32>) outs(%e : tensor<64x64xf32>) permutation = [1, 0]
  return %r : tensor<64x64xf32>
}

// Channel-first NCHW pooling: IREE's strategy pass selects
// ConvTileAndDecomposeExpert; because RVV does not decompose NCHW poolings, it
// preserves the strategy pass's decomposable vector tile sizes.
// CHECK-LABEL: func.func @pooling_nchw_sum_dispatch_0
// CHECK-SAME:    pipeline = #iree_cpu.pipeline<ConvTileAndDecomposeExpert>
// CHECK:         linalg.generic
// CHECK-SAME:      vector_common_parallel = [1, 2, 1, 8, 0, 0], vector_reduction = [0, 0, 0, 0, 1, 3]
func.func @pooling_nchw_sum(%in: tensor<1x16x34x34xf32>) -> tensor<1x16x32x32xf32> {
  %cst = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<1x16x32x32xf32>
  %w = tensor.empty() : tensor<3x3xf32>
  %f = linalg.fill ins(%cst : f32) outs(%e : tensor<1x16x32x32xf32>) -> tensor<1x16x32x32xf32>
  %r = linalg.pooling_nchw_sum {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
       ins(%in, %w : tensor<1x16x34x34xf32>, tensor<3x3xf32>) outs(%f : tensor<1x16x32x32xf32>) -> tensor<1x16x32x32xf32>
  return %r : tensor<1x16x32x32xf32>
}

// Row reduction over a prime extent (197): divisor snapping collapses the
// 16-lane reduction tile to 1 (< 4 lanes), so RVV preserves the strategy
// pass's masked vector tile sizes ([8, 0] and [0, 8]).
// CHECK-LABEL: func.func @reduce_sum_197_dispatch_0
// CHECK:         linalg.generic
// CHECK-SAME:      vector_common_parallel = [8, 0], vector_reduction = [0, 8]
func.func @reduce_sum_197(%x: tensor<64x197xf32>) -> tensor<64xf32> {
  %cst = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<64xf32>
  %f = linalg.fill ins(%cst : f32) outs(%e : tensor<64xf32>) -> tensor<64xf32>
  %r = linalg.reduce ins(%x : tensor<64x197xf32>) outs(%f : tensor<64xf32>) dimensions = [1]
    (%in: f32, %acc: f32) {
      %s = arith.addf %in, %acc : f32
      linalg.yield %s : f32
    }
  return %r : tensor<64xf32>
}

// IREE splits the hand-written softmax into a max-reduction dispatch and an
// exp-sum + normalization dispatch. The latter keeps one-row DTCM tiles
// although its 4096-element rows exceed the budget, and its normalization takes
// the root's 16-lane reduction tile as its inner-parallel tile.
// CHECK-LABEL: func.func @softmax_f32_dispatch_0
// CHECK:         linalg.generic
// CHECK-SAME:      vector_common_parallel = [1, 0], vector_reduction = [0, 16]
// CHECK-LABEL: func.func @softmax_f32_dispatch_1
// CHECK:         linalg.generic
// CHECK-SAME:      distribution = [1, 0], vector_common_parallel = [1, 0], vector_reduction = [0, 16]
// CHECK:         linalg.generic
// CHECK-SAME:      vector_common_parallel = [1, 0], vector_inner_parallel = [0, 16]
func.func @softmax_f32(%in: tensor<8x4096xf32>) -> tensor<8x4096xf32> {
  %neg_inf = arith.constant 0xFF800000 : f32
  %zero = arith.constant 0.0 : f32
  %em = tensor.empty() : tensor<8xf32>
  %fm = linalg.fill ins(%neg_inf : f32) outs(%em : tensor<8xf32>) -> tensor<8xf32>
  %max = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
                         iterator_types = ["parallel", "reduction"]}
      ins(%in : tensor<8x4096xf32>) outs(%fm : tensor<8xf32>) {
  ^bb0(%x: f32, %acc: f32):
    %m = arith.maximumf %x, %acc : f32
    linalg.yield %m : f32
  } -> tensor<8xf32>
  %es = tensor.empty() : tensor<8xf32>
  %fs = linalg.fill ins(%zero : f32) outs(%es : tensor<8xf32>) -> tensor<8xf32>
  %sum = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>],
                         iterator_types = ["parallel", "reduction"]}
      ins(%in, %max : tensor<8x4096xf32>, tensor<8xf32>) outs(%fs : tensor<8xf32>) {
  ^bb0(%x: f32, %m: f32, %acc: f32):
    %d = arith.subf %x, %m : f32
    %ex = math.exp %d : f32
    %s = arith.addf %acc, %ex : f32
    linalg.yield %s : f32
  } -> tensor<8xf32>
  %eo = tensor.empty() : tensor<8x4096xf32>
  %out = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>],
                         iterator_types = ["parallel", "parallel"]}
      ins(%in, %max, %sum : tensor<8x4096xf32>, tensor<8xf32>, tensor<8xf32>) outs(%eo : tensor<8x4096xf32>) {
  ^bb0(%x: f32, %m: f32, %s: f32, %o: f32):
    %d = arith.subf %x, %m : f32
    %ex = math.exp %d : f32
    %q = arith.divf %ex, %s : f32
    linalg.yield %q : f32
  } -> tensor<8x4096xf32>
  return %out : tensor<8x4096xf32>
}

// A streamed reduction loop without a vector tile (vectorSize <= 0) keeps its
// full loop range in the footprint model rather than clamping to 1: with K = 16
// untiled, 1024x16 f32 (64 KB) + output 1024 f32 (4 KB) = 68 KB exceeds 24 KB,
// so the parallel loop shrinks to 256 (16 KB + 1 KB = 17 KB).
// CHECK-LABEL: func.func @reduce_untiled_k_dispatch_0
// CHECK:         linalg.generic
// CHECK-SAME:      lowering_config = #iree_cpu.lowering_config<cache_parallel = [256, 0], cache_reduction = [0, 0], distribution = [256, 0], vector_common_parallel = [1, 0]>
#untiled_k_config = #iree_cpu.lowering_config<distribution = [64, 0],
    vector_common_parallel = [1, 0]>
#untiled_k_compilation = #iree_codegen.compilation_info<
    lowering_config = #untiled_k_config, translation_info = #translation>
func.func @reduce_untiled_k(%x: tensor<1024x16xf32>) -> tensor<1024xf32> {
  %cst = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<1024xf32>
  %f = linalg.fill ins(%cst : f32) outs(%e : tensor<1024xf32>) -> tensor<1024xf32>
  %r = linalg.generic {compilation_info = #untiled_k_compilation,
                       indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
                       iterator_types = ["parallel", "reduction"]}
      ins(%x : tensor<1024x16xf32>) outs(%f : tensor<1024xf32>) {
  ^bb0(%in: f32, %acc: f32):
    %s = arith.addf %in, %acc : f32
    linalg.yield %s : f32
  } -> tensor<1024xf32>
  return %r : tensor<1024xf32>
}

// With M = 196 and a fused consumer, 49 rows exceed 24 KB (res 49x64 f32 +
// output 49x64 f32 + 64x64 f32 scratch > 24 KB), and 196 has no divisor in
// [16, 24]. The search finds the exact divisor 28 in (24, 48] so the Zvt
// workgroup M slice stays static. With 4 KB (SMALL), even minM = 28 at
// nTile = 16 exceeds the budget, so the pass falls back to RVV with promotion.
// CHECK-LABEL: func.func @matmul_zvt_m196_consumer_dispatch_0
// CHECK:         linalg.matmul
// CHECK-SAME:      cache_parallel = [28, 0, 0], cache_reduction = [0, 0, 0], distribution = [28, 0, 0]
// CHECK-SAME:      vector_common_parallel = [16, 0, 0], vector_reduction = [0, 0, 1]
// SMALL-LABEL: func.func @matmul_zvt_m196_consumer_dispatch_0
// SMALL:         linalg.matmul
// SMALL-SAME:      promote_operands = [0, 1, 2]
func.func @matmul_zvt_m196_consumer(%a: tensor<196x64xf32>, %b: tensor<64x64xf32>, %bias: tensor<64xf32>, %res: tensor<196x64xf32>) -> tensor<196x64xf32> {
  %cst = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<196x64xf32>
  %f = linalg.fill ins(%cst : f32) outs(%e : tensor<196x64xf32>) -> tensor<196x64xf32>
  %mm = linalg.matmul ins(%a, %b : tensor<196x64xf32>, tensor<64x64xf32>) outs(%f : tensor<196x64xf32>) -> tensor<196x64xf32>
  %o = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
                       iterator_types = ["parallel", "parallel"]}
      ins(%mm, %bias, %res : tensor<196x64xf32>, tensor<64xf32>, tensor<196x64xf32>) outs(%e : tensor<196x64xf32>) {
  ^bb0(%in: f32, %bi: f32, %re: f32, %out: f32):
    %s1 = arith.addf %in, %bi : f32
    %s2 = arith.addf %s1, %re : f32
    linalg.yield %s2 : f32
  } -> tensor<196x64xf32>
  return %o : tensor<196x64xf32>
}
