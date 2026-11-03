// RUN: iree-compile --mlir-disable-threading --iree-hal-target-device=coralnpu \
// RUN:   --compile-to=executable-configurations %s | FileCheck %s

// Tests Zvt and RVV vector tile size selection, pipeline selection, and
// shape/divisor rules in CoralNPUTileSizeSelectionDTCM under the default 32 KB
// DTCM configuration (24 KB tile budget). Multi-budget, buffer-count, and
// target-feature variations are tested in tile_size_selection_dtcm.mlir.

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
// untiled (RVV caps the 16-lane reduction tile to 0 because it equals the full
// extent), 1024x16 f32 (64 KB) + output 1024 f32 (4 KB) = 68 KB exceeds 24 KB,
// so the parallel loop shrinks to 256 (16 KB + 1 KB = 17 KB).
// CHECK-LABEL: func.func @reduce_untiled_k_dispatch_0
// CHECK:         linalg.generic
// CHECK-SAME:      lowering_config = #iree_cpu.lowering_config<cache_parallel = [256, 0], cache_reduction = [0, 0], distribution = [256, 0], vector_common_parallel = [1, 0]>
func.func @reduce_untiled_k(%x: tensor<1024x16xf32>) -> tensor<1024xf32> {
  %cst = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<1024xf32>
  %f = linalg.fill ins(%cst : f32) outs(%e : tensor<1024xf32>) -> tensor<1024xf32>
  %r = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
                       iterator_types = ["parallel", "reduction"]}
      ins(%x : tensor<1024x16xf32>) outs(%f : tensor<1024xf32>) {
  ^bb0(%in: f32, %acc: f32):
    %s = arith.addf %in, %acc : f32
    linalg.yield %s : f32
  } -> tensor<1024xf32>
  return %r : tensor<1024xf32>
}
