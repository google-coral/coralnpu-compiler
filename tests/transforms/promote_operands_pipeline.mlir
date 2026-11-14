// RUN: %iree_compile --compile-to=executable-targets \
// RUN:   --coralnpu-roofline-speedup-threshold=0 \
// RUN:   --coralnpu-target-cpu-features=+m,+f,+zvl128b,+zve32f %s | FileCheck %s
// RUN: %iree_compile --compile-to=executable-targets \
// RUN:   --coralnpu-roofline-speedup-threshold=0 --coralnpu-dtcm-size-kb=12 \
// RUN:   --coralnpu-target-cpu-features=+m,+f,+zvl128b,+zve32f %s 2>/dev/null \
// RUN:   | FileCheck %s --check-prefix=SMALL
// RUN: %iree_compile --compile-to=executable-targets \
// RUN:   --coralnpu-roofline-speedup-threshold=0 \
// RUN:   --coralnpu-target-cpu-features=+m,+f,+zvl128b,+zve32f \
// RUN:   --mlir-print-ir-after=coralnpu-lower-dtcm-copies %s -o /dev/null 2> %t
// RUN: FileCheck %s --check-prefix=COPIES < %t
// RUN: FileCheck %s --check-prefix=NOGENERIC < %t

// The DTCM promotion (CoralNPUPromoteOperands) through the whole codegen
// pipeline: iree-llvmcpu-assign-workgroup-local-memory packs the DTCM buffers,
// and the export gets their size (workgroup_local_memory). The size must fit in
// the DTCM tile budget: 24 KB for the default 32 KB DTCM (CHECK), and 4 KB for a
// 12 KB DTCM (SMALL). The buffers are the buffers of the footprint model of
// CoralNPUTileSizeSelectionDTCM (see tile_size_selection_dtcm.mlir). The target
// has no Zvt: on a target with Zvt, most of these contractions are Zvt roots,
// which get no promotion.
// CoralNPULowerDTCMCopies lowers the copies between memory and DTCM to loops of
// vector loads and stores (COPIES), and no linalg copy stays (NOGENERIC).

// NOGENERIC-NOT: linalg.generic

// A 4x16 tile with full K: A 4x256 + B 256x16 + C 4x16 f32 = 20736 B. With
// 4 KB, K = 32: A 4x32 + B 32x16 + C 4x16 f32 = 2816 B. The result is stored
// directly, so C is the accumulator and the output staging buffer.
// CHECK-LABEL: hal.executable.export public @matmul_f32_dispatch_0
//       CHECK:   <workgroup_size = [1 : index, 1 : index, 1 : index], workgroup_local_memory = 20736>
// SMALL-LABEL: hal.executable.export public @matmul_f32_dispatch_0
//       SMALL:   <workgroup_size = [1 : index, 1 : index, 1 : index], workgroup_local_memory = 2816>
// The copies of A and B to DTCM, and of the staged C from DTCM: a loop over the
// rows, and vectors of the native vector size (32 B: 8 x f32) along a row.
// COPIES-LABEL: func.func @matmul_f32_dispatch_0
//   COPIES-DAG:   %[[B_DTCM:.+]] = memref.alloc() {{.*}} : memref<256x16xf32, #iree_codegen.workgroup_local>
//   COPIES-DAG:   %[[A_DTCM:.+]] = memref.alloc() {{.*}} : memref<4x256xf32, #iree_codegen.workgroup_local>
//   COPIES-DAG:   %[[C_DTCM:.+]] = memref.alloc() {{.*}} : memref<4x16xf32, #iree_codegen.workgroup_local>
//       COPIES:   %[[C8:.+]] = arith.constant 8 : index
//       COPIES:   scf.for %[[I:[a-z0-9_]+]] = {{.*}} {
//  COPIES-NEXT:     scf.for %[[J:[a-z0-9_]+]] = {{.*}} step %[[C8]] {
//  COPIES-NEXT:       %[[A:.+]] = vector.load %{{.+}}[%[[I]], %[[J]]] : memref<4x256xf32, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>, vector<8xf32>
//  COPIES-NEXT:       vector.store %[[A]], %[[A_DTCM]][%[[I]], %[[J]]] : memref<4x256xf32, #iree_codegen.workgroup_local>, vector<8xf32>
//       COPIES:       %[[B:.+]] = vector.load %{{.+}} : memref<256x16xf32, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>, vector<8xf32>
//  COPIES-NEXT:       vector.store %[[B]], %[[B_DTCM]]{{.*}} : memref<256x16xf32, #iree_codegen.workgroup_local>, vector<8xf32>
//       COPIES:       %[[C:.+]] = vector.load %[[C_DTCM]]{{.*}} : memref<4x16xf32, #iree_codegen.workgroup_local>, vector<8xf32>
//  COPIES-NEXT:       vector.store %[[C]], %{{.+}} : memref<4x16xf32, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>, vector<8xf32>
func.func @matmul_f32(%a: tensor<256x256xf32>, %b: tensor<256x256xf32>) -> tensor<256x256xf32> {
  %cst = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<256x256xf32>
  %f = linalg.fill ins(%cst : f32) outs(%e : tensor<256x256xf32>) -> tensor<256x256xf32>
  %r = linalg.matmul ins(%a, %b : tensor<256x256xf32>, tensor<256x256xf32>) outs(%f : tensor<256x256xf32>) -> tensor<256x256xf32>
  return %r : tensor<256x256xf32>
}

// i8 matmul with a fused bias + truncation: A 32x224 i8 + B 224x64 i8 + bias 64
// i32 + output 32x64 i8 = 23808 B. With 4 KB, K = 56: A 2x56 i8 + B 56x32 i8 +
// bias 32 i32 + output 2x32 i8 = 2096 B. The footprint model also counts the
// split-K accumulator (2x32 i32, 256 B), but the vector loops keep it in
// registers, so its buffer goes away.
// CHECK-LABEL: hal.executable.export public @matmul_bias_trunc_dispatch_0
//       CHECK:   <workgroup_size = [1 : index, 1 : index, 1 : index], workgroup_local_memory = 23808>
// SMALL-LABEL: hal.executable.export public @matmul_bias_trunc_dispatch_0
//       SMALL:   <workgroup_size = [1 : index, 1 : index, 1 : index], workgroup_local_memory = 2096>
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

// Narrow LHS: the i8 LHS is copied to DTCM before the conversion to f32. With
// 24 KB, there is no workgroup loop: A 64x128 i8 + B 128x8 f32 + C 64x8 f32 =
// 14336 B. With 4 KB, a 4x8 tile with K = 64: A 4x64 i8 + B 64x8 f32 + C 4x8
// f32 = 2432 B.
// CHECK-LABEL: hal.executable.export public @matmul_narrow_lhs_dispatch_0
//       CHECK:   <workgroup_size = [1 : index, 1 : index, 1 : index], workgroup_local_memory = 14336>
// SMALL-LABEL: hal.executable.export public @matmul_narrow_lhs_dispatch_0
//       SMALL:   <workgroup_size = [1 : index, 1 : index, 1 : index], workgroup_local_memory = 2432>
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

// bf16 matmul with an f32 accumulator and a fused truncation to bf16 (as after
// CoralNPUPromoteBF16Accumulator): A 2x224 bf16 + B 224x32 bf16 + output 2x32
// bf16 = 15360 B. The f32 accumulator stays in registers.
// CHECK-LABEL: hal.executable.export public @matmul_bf16_trunc_dispatch_0
//       CHECK:   <workgroup_size = [1 : index, 1 : index, 1 : index], workgroup_local_memory = 15360>
func.func @matmul_bf16_trunc(%a: tensor<64x224xbf16>, %b: tensor<224x64xbf16>) -> tensor<64x64xbf16> {
  %cst = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<64x64xf32>
  %f = linalg.fill ins(%cst : f32) outs(%e : tensor<64x64xf32>) -> tensor<64x64xf32>
  %r = linalg.matmul ins(%a, %b : tensor<64x224xbf16>, tensor<224x64xbf16>) outs(%f : tensor<64x64xf32>) -> tensor<64x64xf32>
  %eo = tensor.empty() : tensor<64x64xbf16>
  %o = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
                       iterator_types = ["parallel", "parallel"]}
      ins(%r : tensor<64x64xf32>) outs(%eo : tensor<64x64xbf16>) {
  ^bb0(%in: f32, %out: bf16):
    %t = arith.truncf %in : f32 to bf16
    linalg.yield %t : bf16
  } -> tensor<64x64xbf16>
  return %o : tensor<64x64xbf16>
}

// The same with a 4x32 vector tile (a user config): A 4x224 bf16 + B 224x32
// bf16 + output 4x32 bf16 = 16384 B.
// CHECK-LABEL: hal.executable.export public @matmul_bf16_trunc4_dispatch_0
//       CHECK:   <workgroup_size = [1 : index, 1 : index, 1 : index], workgroup_local_memory = 16384>
#config = #iree_cpu.lowering_config<distribution = [32, 64, 0],
    vector_common_parallel = [4, 32, 0], vector_reduction = [0, 0, 1]>
#translation = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<DoubleTilingExpert>>
#compilation = #iree_codegen.compilation_info<lowering_config = #config,
    translation_info = #translation>
func.func @matmul_bf16_trunc4(%a: tensor<64x224xbf16>, %b: tensor<224x64xbf16>) -> tensor<64x64xbf16> {
  %cst = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<64x64xf32>
  %f = linalg.fill ins(%cst : f32) outs(%e : tensor<64x64xf32>) -> tensor<64x64xf32>
  %r = linalg.matmul {compilation_info = #compilation}
      ins(%a, %b : tensor<64x224xbf16>, tensor<224x64xbf16>) outs(%f : tensor<64x64xf32>) -> tensor<64x64xf32>
  %eo = tensor.empty() : tensor<64x64xbf16>
  %o = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
                       iterator_types = ["parallel", "parallel"]}
      ins(%r : tensor<64x64xf32>) outs(%eo : tensor<64x64xbf16>) {
  ^bb0(%in: f32, %out: bf16):
    %t = arith.truncf %in : f32 to bf16
    linalg.yield %t : bf16
  } -> tensor<64x64xbf16>
  return %o : tensor<64x64xbf16>
}

// Non-final 16-byte-multiple DTCM buffer and inlined constant global: with a
// workgroup loop over M (64 rows in CHECK, 2 rows in SMALL), the promoted
// 32-byte i8 bias buffer is hoisted before B and A and packed at 16-byte
// alignment without 32 B of 64-byte padding (23584 B in CHECK, 2000 B in SMALL),
// and the inlined non-splat constant tensor becomes an alignment = 16
// llvm.mlir.global.
// CHECK-LABEL: hal.executable.export public @matmul_i8_bias_const_dispatch_0
//       CHECK:   <workgroup_size = [1 : index, 1 : index, 1 : index], workgroup_local_memory = 23584>
//       CHECK:   llvm.mlir.global private constant @__constant_32xi8(dense<[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]> : tensor<32xi8>) {addr_space = 0 : i32, alignment = 16 : i64{{.*}}} : !llvm.array<32 x i8>
// SMALL-LABEL: hal.executable.export public @matmul_i8_bias_const_dispatch_0
//       SMALL:   <workgroup_size = [1 : index, 1 : index, 1 : index], workgroup_local_memory = 2000>
func.func @matmul_i8_bias_const(%a: tensor<128x224xi8>, %b: tensor<224x32xi8>, %bias: tensor<32xi8>) -> tensor<128x32xi8> {
  %c0 = arith.constant 0 : i32
  %cst_scale = arith.constant dense<[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]> : tensor<32xi8>
  %e = tensor.empty() : tensor<128x32xi32>
  %f = linalg.fill ins(%c0 : i32) outs(%e : tensor<128x32xi32>) -> tensor<128x32xi32>
  %r = linalg.matmul ins(%a, %b : tensor<128x224xi8>, tensor<224x32xi8>) outs(%f : tensor<128x32xi32>) -> tensor<128x32xi32>
  %eo = tensor.empty() : tensor<128x32xi8>
  %o = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>],
                       iterator_types = ["parallel", "parallel"]}
      ins(%r, %bias, %cst_scale : tensor<128x32xi32>, tensor<32xi8>, tensor<32xi8>) outs(%eo : tensor<128x32xi8>) {
  ^bb0(%in: i32, %bi: i8, %sc: i8, %out: i8):
    %be = arith.extsi %bi : i8 to i32
    %se = arith.extsi %sc : i8 to i32
    %s0 = arith.addi %in, %be : i32
    %s1 = arith.addi %s0, %se : i32
    %t = arith.trunci %s1 : i32 to i8
    linalg.yield %t : i8
  } -> tensor<128x32xi8>
  return %o : tensor<128x32xi8>
}
