// RUN: iree-opt --split-input-file --verify-diagnostics --mlir-print-local-scope \
// RUN:   --remarks-filter-missed=coralnpu-promote-operands \
// RUN:   --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(coralnpu-promote-operands)))))" \
// RUN:   %s | FileCheck %s
// RUN: iree-opt --split-input-file --mlir-print-local-scope \
// RUN:   --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(coralnpu-promote-operands)))))" \
// RUN:   %s 2>/dev/null | FileCheck %s --check-prefix=NOCOPY

// CoralNPUPromoteOperands copies the tiles that the root op (the op with
// promote_operands in its lowering config) and the ops fused with it read to
// DTCM (#iree_codegen.workgroup_local), and stages the stored results in DTCM.
// The DTCM buffers are static bufferization.alloc_tensor ops at the start of
// the workgroup loop body, or of the function if there is no workgroup loop.
// An input tile goes to DTCM with a bufferization.materialize_in_destination
// op. A stored result is computed into a DTCM tensor, and the store op copies
// it to memory when it bufferizes, so the pass makes no copy op for it.
// After bufferization, CoralNPULowerDTCMCopies lowers the copies (see
// lower_dtcm_copies.mlir). Elementwise producers that IREE left out of the tile
// loop are cloned into it first (see producer_out_of_tile_loop).

// The pass makes no linalg copies and no fusion barriers.
// NOCOPY-NOT: linalg.copy
// NOCOPY-NOT: iree_codegen.fusion_barrier

// Split K with a fused bias + truncation: the accumulator, the K chunks of the
// LHS and the RHS, the bias tile, and the output staging buffer are in DTCM.
// The K loop copies its chunks into the same buffers in each iteration.
#target = #hal.executable.target<"coralnpu", "coralnpu-elf", {native_vector_size = 32 : i64}>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, Indirect>
], flags = Indirect>
#translation = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<DoubleTilingExpert>, {enable_loop_peeling}>
#root = #iree_cpu.lowering_config<cache_parallel = [2, 32, 0], cache_reduction = [0, 0, 512], distribution = [2, 32, 0], promote_operands = [0, 1, 2], vector_common_parallel = [2, 32, 0], vector_reduction = [0, 0, 1]>
#elementwise = #iree_cpu.lowering_config<vector_common_parallel = [2, 32]>
// CHECK-LABEL: func.func @split_k_bias_trunc()
//       CHECK:   scf.forall
//  CHECK-NEXT:     %[[ACC:.+]] = bufferization.alloc_tensor() <{memory_space = #iree_codegen.workgroup_local}> : tensor<2x32xi32>
//  CHECK-NEXT:     %[[LHS:.+]] = bufferization.alloc_tensor() <{memory_space = #iree_codegen.workgroup_local}> : tensor<2x512xi8>
//  CHECK-NEXT:     %[[RHS:.+]] = bufferization.alloc_tensor() <{memory_space = #iree_codegen.workgroup_local}> : tensor<512x32xi8>
//  CHECK-NEXT:     %[[BIAS:.+]] = bufferization.alloc_tensor() <{memory_space = #iree_codegen.workgroup_local}> : tensor<32xi32>
//  CHECK-NEXT:     %[[OUT:.+]] = bufferization.alloc_tensor() <{memory_space = #iree_codegen.workgroup_local}> : tensor<2x32xi8>
//       CHECK:     %[[BIAS_TILE:.+]] = tensor.extract_slice
//  CHECK-NEXT:     %[[BIAS_LOCAL:.+]] = bufferization.materialize_in_destination %[[BIAS_TILE]] in %[[BIAS]] : (tensor<32xi32>, tensor<32xi32>) -> tensor<32xi32>
//       CHECK:     %[[FILL:.+]] = linalg.fill {{.*}} outs(%[[ACC]] : tensor<2x32xi32>)
//       CHECK:     %[[K_LOOP:.+]] = scf.for {{.*}} iter_args(%[[ITER:.+]] = %[[FILL]])
//       CHECK:       %[[LHS_TILE:.+]] = tensor.extract_slice
//  CHECK-NEXT:       %[[LHS_LOCAL:.+]] = bufferization.materialize_in_destination %[[LHS_TILE]] in %[[LHS]] : (tensor<2x512xi8>, tensor<2x512xi8>) -> tensor<2x512xi8>
//       CHECK:       %[[RHS_TILE:.+]] = tensor.extract_slice
//  CHECK-NEXT:       %[[RHS_LOCAL:.+]] = bufferization.materialize_in_destination %[[RHS_TILE]] in %[[RHS]] : (tensor<512x32xi8>, tensor<512x32xi8>) -> tensor<512x32xi8>
//       CHECK:       linalg.matmul {{.*}} ins(%[[LHS_LOCAL]], %[[RHS_LOCAL]] : tensor<2x512xi8>, tensor<512x32xi8>) outs(%[[ITER]] : tensor<2x32xi32>)
//       CHECK:     %[[EPILOGUE:.+]] = linalg.generic {{.*}} ins(%[[K_LOOP]], %[[BIAS_LOCAL]] : tensor<2x32xi32>, tensor<32xi32>) outs(%[[OUT]] : tensor<2x32xi8>)
//       CHECK:     scf.forall.in_parallel
//  CHECK-NEXT:       tensor.parallel_insert_slice %[[EPILOGUE]]
hal.executable private @split_k_bias_trunc {
  hal.executable.variant public @coralnpu target(#target) {
    builtin.module {
      func.func @split_k_bias_trunc() attributes {translation_info = #translation} {
        %c512 = arith.constant 512 : index
        %c1024 = arith.constant 1024 : index
        %c0_i32 = arith.constant 0 : i32
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<64x1024xi8, #hal.descriptor_type<storage_buffer>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<1024x64xi8, #hal.descriptor_type<storage_buffer>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<64xi32, #hal.descriptor_type<storage_buffer>>
        %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) alignment(64) offset(%c0) flags(Indirect) : memref<64x64xi8, #hal.descriptor_type<storage_buffer>>
        %4 = iree_codegen.load_from_buffer %0 : memref<64x1024xi8, #hal.descriptor_type<storage_buffer>> -> tensor<64x1024xi8>
        %5 = iree_codegen.load_from_buffer %1 : memref<1024x64xi8, #hal.descriptor_type<storage_buffer>> -> tensor<1024x64xi8>
        %6 = iree_codegen.load_from_buffer %2 : memref<64xi32, #hal.descriptor_type<storage_buffer>> -> tensor<64xi32>
        %7 = tensor.empty() : tensor<64x64xi8>
        %8 = scf.forall (%arg0, %arg1) = (0, 0) to (64, 64) step (2, 32) shared_outs(%arg2 = %7) -> (tensor<64x64xi8>) {
          %9 = tensor.empty() : tensor<2x32xi32>
          %bias = tensor.extract_slice %6[%arg1] [32] [1] : tensor<64xi32> to tensor<32xi32>
          %out = tensor.extract_slice %arg2[%arg0, %arg1] [2, 32] [1, 1] : tensor<64x64xi8> to tensor<2x32xi8>
          %10 = linalg.fill {lowering_config = #elementwise} ins(%c0_i32 : i32) outs(%9 : tensor<2x32xi32>) -> tensor<2x32xi32>
          %11 = scf.for %arg3 = %c0 to %c1024 step %c512 iter_args(%arg4 = %10) -> (tensor<2x32xi32>) {
            %lhs = tensor.extract_slice %4[%arg0, %arg3] [2, 512] [1, 1] : tensor<64x1024xi8> to tensor<2x512xi8>
            %rhs = tensor.extract_slice %5[%arg3, %arg1] [512, 32] [1, 1] : tensor<1024x64xi8> to tensor<512x32xi8>
            %13 = linalg.matmul {lowering_config = #root} ins(%lhs, %rhs : tensor<2x512xi8>, tensor<512x32xi8>) outs(%arg4 : tensor<2x32xi32>) -> tensor<2x32xi32>
            scf.yield %13 : tensor<2x32xi32>
          }
          %12 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%11, %bias : tensor<2x32xi32>, tensor<32xi32>) outs(%out : tensor<2x32xi8>) attrs =  {lowering_config = #elementwise} {
          ^bb0(%in: i32, %in_1: i32, %o: i8):
            %13 = arith.addi %in, %in_1 : i32
            %14 = arith.trunci %13 : i32 to i8
            linalg.yield %14 : i8
          } -> tensor<2x32xi8>
          scf.forall.in_parallel {
            tensor.parallel_insert_slice %12 into %arg2[%arg0, %arg1] [2, 32] [1, 1] : tensor<2x32xi8> into tensor<64x64xi8>
          }
        } {mapping = [#iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]}
        iree_codegen.store_to_buffer %8, %3 : tensor<64x64xi8> into memref<64x64xi8, #hal.descriptor_type<storage_buffer>>
        return
      }
    }
  }
}

// -----

// Full K, and the root result is stored: the accumulator is also the output
// staging buffer. N and K are not tiled, so the root op reads the whole RHS
// load (outside of the workgroup loop): the copy is at the start of the
// workgroup loop body, after the buffers.
#target = #hal.executable.target<"coralnpu", "coralnpu-elf", {native_vector_size = 32 : i64}>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, Indirect>
], flags = Indirect>
#translation = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<DoubleTilingExpert>, {enable_loop_peeling}>
#root = #iree_cpu.lowering_config<cache_parallel = [16, 0, 0], cache_reduction = [0, 0, 0], distribution = [16, 0, 0], promote_operands = [0, 1, 2], vector_common_parallel = [2, 32, 0], vector_reduction = [0, 0, 1]>
#elementwise = #iree_cpu.lowering_config<vector_common_parallel = [2, 32]>
// CHECK-LABEL: func.func @full_k_stored()
//       CHECK:   %[[RHS_LOAD:.+]] = iree_codegen.load_from_buffer {{.*}} -> tensor<64x48xi32>
//       CHECK:   scf.forall
//  CHECK-NEXT:     %[[ACC:.+]] = bufferization.alloc_tensor() <{memory_space = #iree_codegen.workgroup_local}> : tensor<16x48xi32>
//  CHECK-NEXT:     %[[LHS:.+]] = bufferization.alloc_tensor() <{memory_space = #iree_codegen.workgroup_local}> : tensor<16x64xi32>
//  CHECK-NEXT:     %[[RHS:.+]] = bufferization.alloc_tensor() <{memory_space = #iree_codegen.workgroup_local}> : tensor<64x48xi32>
//  CHECK-NEXT:     %[[RHS_LOCAL:.+]] = bufferization.materialize_in_destination %[[RHS_LOAD]] in %[[RHS]] : (tensor<64x48xi32>, tensor<64x48xi32>) -> tensor<64x48xi32>
//       CHECK:     %[[LHS_LOCAL:.+]] = bufferization.materialize_in_destination %{{.+}} in %[[LHS]]
//       CHECK:     %[[FILL:.+]] = linalg.fill {{.*}} outs(%[[ACC]] : tensor<16x48xi32>)
//       CHECK:     %[[MATMUL:.+]] = linalg.matmul {{.*}} ins(%[[LHS_LOCAL]], %[[RHS_LOCAL]] : tensor<16x64xi32>, tensor<64x48xi32>) outs(%[[FILL]] : tensor<16x48xi32>)
//       CHECK:     scf.forall.in_parallel
//  CHECK-NEXT:       tensor.parallel_insert_slice %[[MATMUL]]
hal.executable private @full_k_stored {
  hal.executable.variant public @coralnpu target(#target) {
    builtin.module {
      func.func @full_k_stored() attributes {translation_info = #translation} {
        %c0_i32 = arith.constant 0 : i32
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<32x64xi32, #hal.descriptor_type<storage_buffer>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<64x48xi32, #hal.descriptor_type<storage_buffer>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(Indirect) : memref<32x48xi32, #hal.descriptor_type<storage_buffer>>
        %3 = iree_codegen.load_from_buffer %0 : memref<32x64xi32, #hal.descriptor_type<storage_buffer>> -> tensor<32x64xi32>
        %4 = iree_codegen.load_from_buffer %1 : memref<64x48xi32, #hal.descriptor_type<storage_buffer>> -> tensor<64x48xi32>
        %5 = tensor.empty() : tensor<32x48xi32>
        %6 = scf.forall (%arg0) = (0) to (32) step (16) shared_outs(%arg1 = %5) -> (tensor<32x48xi32>) {
          %out = tensor.extract_slice %arg1[%arg0, 0] [16, 48] [1, 1] : tensor<32x48xi32> to tensor<16x48xi32>
          %lhs = tensor.extract_slice %3[%arg0, 0] [16, 64] [1, 1] : tensor<32x64xi32> to tensor<16x64xi32>
          %7 = linalg.fill {lowering_config = #elementwise} ins(%c0_i32 : i32) outs(%out : tensor<16x48xi32>) -> tensor<16x48xi32>
          %8 = linalg.matmul {lowering_config = #root} ins(%lhs, %4 : tensor<16x64xi32>, tensor<64x48xi32>) outs(%7 : tensor<16x48xi32>) -> tensor<16x48xi32>
          scf.forall.in_parallel {
            tensor.parallel_insert_slice %8 into %arg1[%arg0, 0] [16, 48] [1, 1] : tensor<16x48xi32> into tensor<32x48xi32>
          }
        } {mapping = [#iree_codegen.workgroup_mapping<x>]}
        iree_codegen.store_to_buffer %6, %2 : tensor<32x48xi32> into memref<32x48xi32, #hal.descriptor_type<storage_buffer>>
        return
      }
    }
  }
}

// -----

// Partial tiles: N = 450 is not a multiple of the tile size 32, so the
// workgroup tile and the inner (cache-level) loop have dynamic sizes. The DTCM
// buffers have the static upper bounds of the tile sizes, and the dynamic tiles
// use slices of them. materialize_in_destination is a destination-style op, so
// the sizes of its result are the slice sizes (%N), which peeling can make
// static.
#target = #hal.executable.target<"coralnpu", "coralnpu-elf", {native_vector_size = 32 : i64}>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, Indirect>
], flags = Indirect>
#translation = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<DoubleTilingExpert>, {enable_loop_peeling}>
#root = #iree_cpu.lowering_config<cache_parallel = [2, 32, 0], cache_reduction = [0, 0, 0], distribution = [2, 32, 0], promote_operands = [0, 1, 2], vector_common_parallel = [2, 32, 0], vector_reduction = [0, 0, 1]>
#elementwise = #iree_cpu.lowering_config<vector_common_parallel = [2, 32]>
// CHECK-LABEL: func.func @partial_tiles()
//       CHECK:   scf.forall
//  CHECK-NEXT:     %[[ACC:.+]] = bufferization.alloc_tensor() <{memory_space = #iree_codegen.workgroup_local}> : tensor<2x32xi32>
//  CHECK-NEXT:     %[[LHS:.+]] = bufferization.alloc_tensor() <{memory_space = #iree_codegen.workgroup_local}> : tensor<2x100xi32>
//  CHECK-NEXT:     %[[RHS:.+]] = bufferization.alloc_tensor() <{memory_space = #iree_codegen.workgroup_local}> : tensor<100x32xi32>
//       CHECK:     scf.forall
//       CHECK:       %[[N:.+]] = affine.min
//       CHECK:       %[[LHS_LOCAL:.+]] = bufferization.materialize_in_destination %{{.+}} in %[[LHS]] : (tensor<2x100xi32>, tensor<2x100xi32>) -> tensor<2x100xi32>
//       CHECK:       %[[RHS_TILE:.+]] = tensor.extract_slice %{{.+}}[0, %{{.+}}] [100, %[[N]]] [1, 1]
//  CHECK-NEXT:       %[[RHS_SLICE:.+]] = tensor.extract_slice %[[RHS]][0, 0] [100, %[[N]]] [1, 1] : tensor<100x32xi32> to tensor<100x?xi32>
//  CHECK-NEXT:       %[[RHS_LOCAL:.+]] = bufferization.materialize_in_destination %[[RHS_TILE]] in %[[RHS_SLICE]] : (tensor<100x?xi32>, tensor<100x?xi32>) -> tensor<100x?xi32>
//       CHECK:       %[[ACC_SLICE:.+]] = tensor.extract_slice %[[ACC]][0, 0] [2, %[[N]]] [1, 1] : tensor<2x32xi32> to tensor<2x?xi32>
//       CHECK:       %[[FILL:.+]] = linalg.fill {{.*}} outs(%[[ACC_SLICE]] : tensor<2x?xi32>)
//       CHECK:       %[[MATMUL:.+]] = linalg.matmul {{.*}} ins(%[[LHS_LOCAL]], %[[RHS_LOCAL]] : tensor<2x100xi32>, tensor<100x?xi32>) outs(%[[FILL]] : tensor<2x?xi32>)
//       CHECK:       scf.forall.in_parallel
//  CHECK-NEXT:         tensor.parallel_insert_slice %[[MATMUL]]
hal.executable private @partial_tiles {
  hal.executable.variant public @coralnpu target(#target) {
    builtin.module {
      func.func @partial_tiles() attributes {translation_info = #translation} {
        %c0_i32 = arith.constant 0 : i32
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<300x100xi32, #hal.descriptor_type<storage_buffer>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<100x450xi32, #hal.descriptor_type<storage_buffer>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(Indirect) : memref<300x450xi32, #hal.descriptor_type<storage_buffer>>
        %3 = iree_codegen.load_from_buffer %0 : memref<300x100xi32, #hal.descriptor_type<storage_buffer>> -> tensor<300x100xi32>
        %4 = iree_codegen.load_from_buffer %1 : memref<100x450xi32, #hal.descriptor_type<storage_buffer>> -> tensor<100x450xi32>
        %5 = tensor.empty() : tensor<300x450xi32>
        %6 = scf.forall (%arg0, %arg1) = (0, 0) to (300, 450) step (2, 32) shared_outs(%arg2 = %5) -> (tensor<300x450xi32>) {
          %7 = affine.min affine_map<(d0) -> (-d0 + 450, 32)>(%arg1)
          %wg_out = tensor.extract_slice %arg2[%arg0, %arg1] [2, %7] [1, 1] : tensor<300x450xi32> to tensor<2x?xi32>
          %8 = scf.forall (%arg3) = (0) to (%7) step (32) shared_outs(%arg4 = %wg_out) -> (tensor<2x?xi32>) {
            %9 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 32)>(%arg3)[%7]
            %lhs = tensor.extract_slice %3[%arg0, 0] [2, 100] [1, 1] : tensor<300x100xi32> to tensor<2x100xi32>
            %10 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%arg3, %arg1]
            %rhs = tensor.extract_slice %4[0, %10] [100, %9] [1, 1] : tensor<100x450xi32> to tensor<100x?xi32>
            %out = tensor.extract_slice %arg4[0, %arg3] [2, %9] [1, 1] : tensor<2x?xi32> to tensor<2x?xi32>
            %11 = linalg.fill {lowering_config = #elementwise} ins(%c0_i32 : i32) outs(%out : tensor<2x?xi32>) -> tensor<2x?xi32>
            %12 = linalg.matmul {lowering_config = #root} ins(%lhs, %rhs : tensor<2x100xi32>, tensor<100x?xi32>) outs(%11 : tensor<2x?xi32>) -> tensor<2x?xi32>
            scf.forall.in_parallel {
              tensor.parallel_insert_slice %12 into %arg4[0, %arg3] [2, %9] [1, 1] : tensor<2x?xi32> into tensor<2x?xi32>
            }
          }
          scf.forall.in_parallel {
            tensor.parallel_insert_slice %8 into %arg2[%arg0, %arg1] [2, %7] [1, 1] : tensor<2x?xi32> into tensor<300x450xi32>
          }
        } {mapping = [#iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]}
        iree_codegen.store_to_buffer %6, %2 : tensor<300x450xi32> into memref<300x450xi32, #hal.descriptor_type<storage_buffer>>
        return
      }
    }
  }
}

// -----

// No workgroup loop (the whole problem fits in DTCM): the DTCM buffers are at
// the start of the function. The narrow i8 LHS is copied before the fused
// conversion to f32, and store_to_buffer copies the staged result.
#target = #hal.executable.target<"coralnpu", "coralnpu-elf", {native_vector_size = 32 : i64}>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, Indirect>
], flags = Indirect>
#translation = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<DoubleTilingExpert>, {enable_loop_peeling}>
#root = #iree_cpu.lowering_config<cache_parallel = [0, 0, 0], cache_reduction = [0, 0, 0], distribution = [0, 0, 0], promote_operands = [0, 1, 2], vector_common_parallel = [4, 16, 0], vector_reduction = [0, 0, 1]>
#producer = #iree_cpu.lowering_config<vector_common_parallel = [4, 0], vector_inner_parallel = [0, 8], vector_reduction = [0, 1]>
#elementwise = #iree_cpu.lowering_config<vector_common_parallel = [4, 16]>
// CHECK-LABEL: func.func @no_workgroup_loop()
//  CHECK-NEXT:   %[[ACC:.+]] = bufferization.alloc_tensor() <{memory_space = #iree_codegen.workgroup_local}> : tensor<64x8xf32>
//  CHECK-NEXT:   %[[LHS:.+]] = bufferization.alloc_tensor() <{memory_space = #iree_codegen.workgroup_local}> : tensor<64x128xi8>
//  CHECK-NEXT:   %[[RHS:.+]] = bufferization.alloc_tensor() <{memory_space = #iree_codegen.workgroup_local}> : tensor<128x8xf32>
//       CHECK:   %[[LHS_LOAD:.+]] = iree_codegen.load_from_buffer {{.*}} -> tensor<64x128xi8>
//  CHECK-NEXT:   %[[LHS_LOCAL:.+]] = bufferization.materialize_in_destination %[[LHS_LOAD]] in %[[LHS]] : (tensor<64x128xi8>, tensor<64x128xi8>) -> tensor<64x128xi8>
//       CHECK:   %[[RHS_LOAD:.+]] = iree_codegen.load_from_buffer {{.*}} -> tensor<128x8xf32>
//  CHECK-NEXT:   %[[RHS_LOCAL:.+]] = bufferization.materialize_in_destination %[[RHS_LOAD]] in %[[RHS]] : (tensor<128x8xf32>, tensor<128x8xf32>) -> tensor<128x8xf32>
//       CHECK:   %[[LHS_F32:.+]] = linalg.generic {{.*}} ins(%[[LHS_LOCAL]] : tensor<64x128xi8>)
//       CHECK:   %[[FILL:.+]] = linalg.fill {{.*}} outs(%[[ACC]] : tensor<64x8xf32>)
//       CHECK:   %[[MATMUL:.+]] = linalg.matmul {{.*}} ins(%[[LHS_F32]], %[[RHS_LOCAL]] : tensor<64x128xf32>, tensor<128x8xf32>) outs(%[[FILL]] : tensor<64x8xf32>)
//  CHECK-NEXT:   iree_codegen.store_to_buffer %[[MATMUL]]
hal.executable private @no_workgroup_loop {
  hal.executable.variant public @coralnpu target(#target) {
    builtin.module {
      func.func @no_workgroup_loop() attributes {translation_info = #translation} {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<64x128xi8, #hal.descriptor_type<storage_buffer>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<128x8xf32, #hal.descriptor_type<storage_buffer>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(Indirect) : memref<64x8xf32, #hal.descriptor_type<storage_buffer>>
        %3 = iree_codegen.load_from_buffer %0 : memref<64x128xi8, #hal.descriptor_type<storage_buffer>> -> tensor<64x128xi8>
        %4 = iree_codegen.load_from_buffer %1 : memref<128x8xf32, #hal.descriptor_type<storage_buffer>> -> tensor<128x8xf32>
        %5 = tensor.empty() : tensor<64x8xf32>
        %6 = tensor.empty() : tensor<64x128xf32>
        %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3 : tensor<64x128xi8>) outs(%6 : tensor<64x128xf32>) attrs =  {lowering_config = #producer} {
        ^bb0(%in: i8, %out: f32):
          %10 = arith.sitofp %in : i8 to f32
          linalg.yield %10 : f32
        } -> tensor<64x128xf32>
        %8 = linalg.fill {lowering_config = #elementwise} ins(%cst : f32) outs(%5 : tensor<64x8xf32>) -> tensor<64x8xf32>
        %9 = linalg.matmul {lowering_config = #root} ins(%7, %4 : tensor<64x128xf32>, tensor<128x8xf32>) outs(%8 : tensor<64x8xf32>) -> tensor<64x8xf32>
        iree_codegen.store_to_buffer %9, %2 : tensor<64x8xf32> into memref<64x8xf32, #hal.descriptor_type<storage_buffer>>
        return
      }
    }
  }
}

// -----

// A producer out of the tile loop: only M is tiled, so IREE leaves the
// dequantization of the RHS (K x N) before the workgroup loop. The pass clones
// it into the loop and erases the original, so that its i8 input gets a DTCM
// copy like the inputs of the producers that IREE fuses.
#target = #hal.executable.target<"coralnpu", "coralnpu-elf", {native_vector_size = 32 : i64}>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, Indirect>
], flags = Indirect>
#translation = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<DoubleTilingExpert>, {enable_loop_peeling}>
#root = #iree_cpu.lowering_config<cache_parallel = [8, 0, 0], cache_reduction = [0, 0, 0], distribution = [8, 0, 0], promote_operands = [0, 1, 2], vector_common_parallel = [4, 16, 0], vector_reduction = [0, 0, 1]>
#elementwise = #iree_cpu.lowering_config<vector_common_parallel = [4, 16]>
// CHECK-LABEL: func.func @producer_out_of_tile_loop()
//       CHECK:   %[[RHS_LOAD:.+]] = iree_codegen.load_from_buffer {{.*}} -> tensor<48x200xi8>
//   CHECK-NOT:   linalg.generic
//       CHECK:   scf.forall
//  CHECK-NEXT:     %[[ACC:.+]] = bufferization.alloc_tensor() <{memory_space = #iree_codegen.workgroup_local}> : tensor<8x200xf32>
//  CHECK-NEXT:     %[[RHS:.+]] = bufferization.alloc_tensor() <{memory_space = #iree_codegen.workgroup_local}> : tensor<48x200xi8>
//  CHECK-NEXT:     %[[RHS_LOCAL:.+]] = bufferization.materialize_in_destination %[[RHS_LOAD]] in %[[RHS]] : (tensor<48x200xi8>, tensor<48x200xi8>) -> tensor<48x200xi8>
//  CHECK-NEXT:     %[[LHS:.+]] = bufferization.alloc_tensor() <{memory_space = #iree_codegen.workgroup_local}> : tensor<8x48xf32>
//  CHECK-NEXT:     %[[RHS_F32:.+]] = linalg.generic {{.*}} ins(%[[RHS_LOCAL]] : tensor<48x200xi8>)
//       CHECK:     %[[LHS_LOCAL:.+]] = bufferization.materialize_in_destination %{{.+}} in %[[LHS]]
//       CHECK:     %[[FILL:.+]] = linalg.fill {{.*}} outs(%[[ACC]] : tensor<8x200xf32>)
//       CHECK:     linalg.matmul {{.*}} ins(%[[LHS_LOCAL]], %[[RHS_F32]] : tensor<8x48xf32>, tensor<48x200xf32>) outs(%[[FILL]] : tensor<8x200xf32>)
hal.executable private @producer_out_of_tile_loop {
  hal.executable.variant public @coralnpu target(#target) {
    builtin.module {
      func.func @producer_out_of_tile_loop() attributes {translation_info = #translation} {
        %cst = arith.constant 0.000000e+00 : f32
        %scale = arith.constant 2.500000e-01 : f32
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<48x200xi8, #hal.descriptor_type<storage_buffer>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<40x48xf32, #hal.descriptor_type<storage_buffer>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(Indirect) : memref<40x200xf32, #hal.descriptor_type<storage_buffer>>
        %3 = iree_codegen.load_from_buffer %0 : memref<48x200xi8, #hal.descriptor_type<storage_buffer>> -> tensor<48x200xi8>
        %4 = iree_codegen.load_from_buffer %1 : memref<40x48xf32, #hal.descriptor_type<storage_buffer>> -> tensor<40x48xf32>
        %5 = tensor.empty() : tensor<40x200xf32>
        %6 = tensor.empty() : tensor<48x200xf32>
        %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3 : tensor<48x200xi8>) outs(%6 : tensor<48x200xf32>) attrs = {lowering_config = #elementwise} {
        ^bb0(%in: i8, %out: f32):
          %f = arith.sitofp %in : i8 to f32
          %s = arith.mulf %f, %scale : f32
          linalg.yield %s : f32
        } -> tensor<48x200xf32>
        %8 = scf.forall (%arg0) = (0) to (40) step (8) shared_outs(%arg1 = %5) -> (tensor<40x200xf32>) {
          %out = tensor.extract_slice %arg1[%arg0, 0] [8, 200] [1, 1] : tensor<40x200xf32> to tensor<8x200xf32>
          %lhs = tensor.extract_slice %4[%arg0, 0] [8, 48] [1, 1] : tensor<40x48xf32> to tensor<8x48xf32>
          %9 = linalg.fill {lowering_config = #elementwise} ins(%cst : f32) outs(%out : tensor<8x200xf32>) -> tensor<8x200xf32>
          %10 = linalg.matmul {lowering_config = #root} ins(%lhs, %7 : tensor<8x48xf32>, tensor<48x200xf32>) outs(%9 : tensor<8x200xf32>) -> tensor<8x200xf32>
          scf.forall.in_parallel {
            tensor.parallel_insert_slice %10 into %arg1[%arg0, 0] [8, 200] [1, 1] : tensor<8x200xf32> into tensor<40x200xf32>
          }
        } {mapping = [#iree_codegen.workgroup_mapping<x>]}
        iree_codegen.store_to_buffer %8, %2 : tensor<40x200xf32> into memref<40x200xf32, #hal.descriptor_type<storage_buffer>>
        return
      }
    }
  }
}

// -----

// A fused consumer with two stored results: each result has its own output
// staging buffer. The RHS load comes from outside of the workgroup loop, so
// its copy is right after its buffer, before the other buffers.
#target = #hal.executable.target<"coralnpu", "coralnpu-elf", {native_vector_size = 32 : i64}>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, Indirect>,
  #hal.pipeline.binding<storage_buffer, Indirect>
], flags = Indirect>
#translation = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<DoubleTilingExpert>, {enable_loop_peeling}>
#root = #iree_cpu.lowering_config<cache_parallel = [4, 0, 0], cache_reduction = [0, 0, 0], distribution = [4, 0, 0], promote_operands = [0, 1, 2], vector_common_parallel = [4, 16, 0], vector_reduction = [0, 0, 1]>
#elementwise = #iree_cpu.lowering_config<vector_common_parallel = [4, 16]>
// CHECK-LABEL: func.func @two_stored_results()
//       CHECK:   scf.forall
//  CHECK-NEXT:     %[[LHS:.+]] = bufferization.alloc_tensor() <{memory_space = #iree_codegen.workgroup_local}> : tensor<4x32xf32>
//  CHECK-NEXT:     %[[RHS:.+]] = bufferization.alloc_tensor() <{memory_space = #iree_codegen.workgroup_local}> : tensor<32x16xf32>
//  CHECK-NEXT:     %[[RHS_LOCAL:.+]] = bufferization.materialize_in_destination %{{.+}} in %[[RHS]]
//  CHECK-NEXT:     %[[OUT0:.+]] = bufferization.alloc_tensor() <{memory_space = #iree_codegen.workgroup_local}> : tensor<4x16xf32>
//  CHECK-NEXT:     %[[OUT1:.+]] = bufferization.alloc_tensor() <{memory_space = #iree_codegen.workgroup_local}> : tensor<4x16xf32>
//       CHECK:     %[[MATMUL:.+]] = linalg.matmul {{.*}} ins(%{{.+}}, %[[RHS_LOCAL]] : tensor<4x32xf32>, tensor<32x16xf32>)
//       CHECK:     %[[RESULTS:.+]]:2 = linalg.generic {{.*}} ins(%[[MATMUL]] : tensor<4x16xf32>) outs(%[[OUT0]], %[[OUT1]] : tensor<4x16xf32>, tensor<4x16xf32>)
//       CHECK:     scf.forall.in_parallel
//  CHECK-NEXT:       tensor.parallel_insert_slice %[[RESULTS]]#0
//  CHECK-NEXT:       tensor.parallel_insert_slice %[[RESULTS]]#1
hal.executable private @two_stored_results {
  hal.executable.variant public @coralnpu target(#target) {
    builtin.module {
      func.func @two_stored_results() attributes {translation_info = #translation} {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<8x32xf32, #hal.descriptor_type<storage_buffer>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<32x16xf32, #hal.descriptor_type<storage_buffer>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(Indirect) : memref<8x16xf32, #hal.descriptor_type<storage_buffer>>
        %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) alignment(64) offset(%c0) flags(Indirect) : memref<8x16xf32, #hal.descriptor_type<storage_buffer>>
        %4 = iree_codegen.load_from_buffer %0 : memref<8x32xf32, #hal.descriptor_type<storage_buffer>> -> tensor<8x32xf32>
        %5 = iree_codegen.load_from_buffer %1 : memref<32x16xf32, #hal.descriptor_type<storage_buffer>> -> tensor<32x16xf32>
        %6 = tensor.empty() : tensor<8x16xf32>
        %7:2 = scf.forall (%arg0) = (0) to (8) step (4) shared_outs(%arg1 = %6, %arg2 = %6) -> (tensor<8x16xf32>, tensor<8x16xf32>) {
          %lhs = tensor.extract_slice %4[%arg0, 0] [4, 32] [1, 1] : tensor<8x32xf32> to tensor<4x32xf32>
          %e = tensor.empty() : tensor<4x16xf32>
          %f = linalg.fill {lowering_config = #elementwise} ins(%cst : f32) outs(%e : tensor<4x16xf32>) -> tensor<4x16xf32>
          %m = linalg.matmul {lowering_config = #root} ins(%lhs, %5 : tensor<4x32xf32>, tensor<32x16xf32>) outs(%f : tensor<4x16xf32>) -> tensor<4x16xf32>
          %out0 = tensor.extract_slice %arg1[%arg0, 0] [4, 16] [1, 1] : tensor<8x16xf32> to tensor<4x16xf32>
          %out1 = tensor.extract_slice %arg2[%arg0, 0] [4, 16] [1, 1] : tensor<8x16xf32> to tensor<4x16xf32>
          %r:2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%m : tensor<4x16xf32>) outs(%out0, %out1 : tensor<4x16xf32>, tensor<4x16xf32>) attrs = {lowering_config = #elementwise} {
          ^bb0(%in: f32, %o0: f32, %o1: f32):
            %neg = arith.negf %in : f32
            linalg.yield %in, %neg : f32, f32
          } -> (tensor<4x16xf32>, tensor<4x16xf32>)
          scf.forall.in_parallel {
            tensor.parallel_insert_slice %r#0 into %arg1[%arg0, 0] [4, 16] [1, 1] : tensor<4x16xf32> into tensor<8x16xf32>
            tensor.parallel_insert_slice %r#1 into %arg2[%arg0, 0] [4, 16] [1, 1] : tensor<4x16xf32> into tensor<8x16xf32>
          }
        } {mapping = [#iree_codegen.workgroup_mapping<x>]}
        iree_codegen.store_to_buffer %7#0, %2 : tensor<8x16xf32> into memref<8x16xf32, #hal.descriptor_type<storage_buffer>>
        iree_codegen.store_to_buffer %7#1, %3 : tensor<8x16xf32> into memref<8x16xf32, #hal.descriptor_type<storage_buffer>>
        return
      }
    }
  }
}

// -----

// An init that the root op reads and that is not a fill (a broadcast of a
// bias): the broadcast reads a DTCM copy of the bias, and its result is copied
// to the DTCM accumulator, which is also the output staging buffer (the root
// result is stored).
#target = #hal.executable.target<"coralnpu", "coralnpu-elf", {native_vector_size = 32 : i64}>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, Indirect>
], flags = Indirect>
#translation = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<DoubleTilingExpert>, {enable_loop_peeling}>
#root = #iree_cpu.lowering_config<cache_parallel = [4, 0, 0], cache_reduction = [0, 0, 0], distribution = [4, 0, 0], promote_operands = [0, 1, 2], vector_common_parallel = [4, 16, 0], vector_reduction = [0, 0, 1]>
#elementwise = #iree_cpu.lowering_config<vector_common_parallel = [4, 16]>
// CHECK-LABEL: func.func @broadcast_bias_init()
//       CHECK:   %[[BIAS_LOAD:.+]] = iree_codegen.load_from_buffer {{.*}} -> tensor<16xf32>
//       CHECK:   scf.forall
//  CHECK-NEXT:     %[[ACC:.+]] = bufferization.alloc_tensor() <{memory_space = #iree_codegen.workgroup_local}> : tensor<4x16xf32>
//  CHECK-NEXT:     %[[BIAS:.+]] = bufferization.alloc_tensor() <{memory_space = #iree_codegen.workgroup_local}> : tensor<16xf32>
//  CHECK-NEXT:     %[[BIAS_LOCAL:.+]] = bufferization.materialize_in_destination %[[BIAS_LOAD]] in %[[BIAS]] : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
//       CHECK:     %[[BCAST:.+]] = linalg.generic {{.*}} ins(%[[BIAS_LOCAL]] : tensor<16xf32>)
//       CHECK:     %[[ACC_INIT:.+]] = bufferization.materialize_in_destination %[[BCAST]] in %[[ACC]] : (tensor<4x16xf32>, tensor<4x16xf32>) -> tensor<4x16xf32>
//  CHECK-NEXT:     %[[MATMUL:.+]] = linalg.matmul {{.*}} outs(%[[ACC_INIT]] : tensor<4x16xf32>)
//       CHECK:     scf.forall.in_parallel
//  CHECK-NEXT:       tensor.parallel_insert_slice %[[MATMUL]]
hal.executable private @broadcast_bias_init {
  hal.executable.variant public @coralnpu target(#target) {
    builtin.module {
      func.func @broadcast_bias_init() attributes {translation_info = #translation} {
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<8x32xf32, #hal.descriptor_type<storage_buffer>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<32x16xf32, #hal.descriptor_type<storage_buffer>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<16xf32, #hal.descriptor_type<storage_buffer>>
        %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) alignment(64) offset(%c0) flags(Indirect) : memref<8x16xf32, #hal.descriptor_type<storage_buffer>>
        %4 = iree_codegen.load_from_buffer %0 : memref<8x32xf32, #hal.descriptor_type<storage_buffer>> -> tensor<8x32xf32>
        %5 = iree_codegen.load_from_buffer %1 : memref<32x16xf32, #hal.descriptor_type<storage_buffer>> -> tensor<32x16xf32>
        %6 = iree_codegen.load_from_buffer %2 : memref<16xf32, #hal.descriptor_type<storage_buffer>> -> tensor<16xf32>
        %7 = tensor.empty() : tensor<8x16xf32>
        %8 = scf.forall (%arg0) = (0) to (8) step (4) shared_outs(%arg1 = %7) -> (tensor<8x16xf32>) {
          %lhs = tensor.extract_slice %4[%arg0, 0] [4, 32] [1, 1] : tensor<8x32xf32> to tensor<4x32xf32>
          %e = tensor.empty() : tensor<4x16xf32>
          %b = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%6 : tensor<16xf32>) outs(%e : tensor<4x16xf32>) attrs = {lowering_config = #elementwise} {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          } -> tensor<4x16xf32>
          %m = linalg.matmul {lowering_config = #root} ins(%lhs, %5 : tensor<4x32xf32>, tensor<32x16xf32>) outs(%b : tensor<4x16xf32>) -> tensor<4x16xf32>
          scf.forall.in_parallel {
            tensor.parallel_insert_slice %m into %arg1[%arg0, 0] [4, 16] [1, 1] : tensor<4x16xf32> into tensor<8x16xf32>
          }
        } {mapping = [#iree_codegen.workgroup_mapping<x>]}
        iree_codegen.store_to_buffer %8, %3 : tensor<8x16xf32> into memref<8x16xf32, #hal.descriptor_type<storage_buffer>>
        return
      }
    }
  }
}

// -----

// A fill with two users: the root op gets a clone of the fill that writes into
// the DTCM accumulator, and the other user keeps the old fill.
#target = #hal.executable.target<"coralnpu", "coralnpu-elf", {native_vector_size = 32 : i64}>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, Indirect>,
  #hal.pipeline.binding<storage_buffer, Indirect>
], flags = Indirect>
#translation = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<DoubleTilingExpert>, {enable_loop_peeling}>
#root = #iree_cpu.lowering_config<cache_parallel = [4, 0, 0], cache_reduction = [0, 0, 0], distribution = [4, 0, 0], promote_operands = [0, 1, 2], vector_common_parallel = [4, 16, 0], vector_reduction = [0, 0, 1]>
#elementwise = #iree_cpu.lowering_config<vector_common_parallel = [4, 16]>
// CHECK-LABEL: func.func @shared_fill()
//       CHECK:   scf.forall
//  CHECK-NEXT:     %[[ACC:.+]] = bufferization.alloc_tensor() <{memory_space = #iree_codegen.workgroup_local}> : tensor<4x16xf32>
//       CHECK:     %[[OUT1:.+]] = bufferization.alloc_tensor() <{memory_space = #iree_codegen.workgroup_local}> : tensor<4x16xf32>
//       CHECK:     %[[EMPTY:.+]] = tensor.empty() : tensor<4x16xf32>
//  CHECK-NEXT:     %[[ACC_FILL:.+]] = linalg.fill {{.*}} outs(%[[ACC]] : tensor<4x16xf32>)
//  CHECK-NEXT:     %[[OLD_FILL:.+]] = linalg.fill {{.*}} outs(%[[EMPTY]] : tensor<4x16xf32>)
//  CHECK-NEXT:     %[[MATMUL:.+]] = linalg.matmul {{.*}} outs(%[[ACC_FILL]] : tensor<4x16xf32>)
//       CHECK:     %[[NEG:.+]] = linalg.generic {{.*}} ins(%[[OLD_FILL]] : tensor<4x16xf32>) outs(%[[OUT1]] : tensor<4x16xf32>)
//       CHECK:     scf.forall.in_parallel
//  CHECK-NEXT:       tensor.parallel_insert_slice %[[MATMUL]]
//  CHECK-NEXT:       tensor.parallel_insert_slice %[[NEG]]
hal.executable private @shared_fill {
  hal.executable.variant public @coralnpu target(#target) {
    builtin.module {
      func.func @shared_fill() attributes {translation_info = #translation} {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<8x32xf32, #hal.descriptor_type<storage_buffer>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<32x16xf32, #hal.descriptor_type<storage_buffer>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(Indirect) : memref<8x16xf32, #hal.descriptor_type<storage_buffer>>
        %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) alignment(64) offset(%c0) flags(Indirect) : memref<8x16xf32, #hal.descriptor_type<storage_buffer>>
        %4 = iree_codegen.load_from_buffer %0 : memref<8x32xf32, #hal.descriptor_type<storage_buffer>> -> tensor<8x32xf32>
        %5 = iree_codegen.load_from_buffer %1 : memref<32x16xf32, #hal.descriptor_type<storage_buffer>> -> tensor<32x16xf32>
        %6 = tensor.empty() : tensor<8x16xf32>
        %7:2 = scf.forall (%arg0) = (0) to (8) step (4) shared_outs(%arg1 = %6, %arg2 = %6) -> (tensor<8x16xf32>, tensor<8x16xf32>) {
          %lhs = tensor.extract_slice %4[%arg0, 0] [4, 32] [1, 1] : tensor<8x32xf32> to tensor<4x32xf32>
          %e = tensor.empty() : tensor<4x16xf32>
          %f = linalg.fill {lowering_config = #elementwise} ins(%cst : f32) outs(%e : tensor<4x16xf32>) -> tensor<4x16xf32>
          %m = linalg.matmul {lowering_config = #root} ins(%lhs, %5 : tensor<4x32xf32>, tensor<32x16xf32>) outs(%f : tensor<4x16xf32>) -> tensor<4x16xf32>
          %out1 = tensor.extract_slice %arg2[%arg0, 0] [4, 16] [1, 1] : tensor<8x16xf32> to tensor<4x16xf32>
          %r = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%f : tensor<4x16xf32>) outs(%out1 : tensor<4x16xf32>) attrs = {lowering_config = #elementwise} {
          ^bb0(%in: f32, %o: f32):
            %neg = arith.negf %in : f32
            linalg.yield %neg : f32
          } -> tensor<4x16xf32>
          scf.forall.in_parallel {
            tensor.parallel_insert_slice %m into %arg1[%arg0, 0] [4, 16] [1, 1] : tensor<4x16xf32> into tensor<8x16xf32>
            tensor.parallel_insert_slice %r into %arg2[%arg0, 0] [4, 16] [1, 1] : tensor<4x16xf32> into tensor<8x16xf32>
          }
        } {mapping = [#iree_codegen.workgroup_mapping<x>]}
        iree_codegen.store_to_buffer %7#0, %2 : tensor<8x16xf32> into memref<8x16xf32, #hal.descriptor_type<storage_buffer>>
        iree_codegen.store_to_buffer %7#1, %3 : tensor<8x16xf32> into memref<8x16xf32, #hal.descriptor_type<storage_buffer>>
        return
      }
    }
  }
}

// -----

// Tiles without a static size bound stay in memory, with a missed remark. The
// RHS tile has static sizes, so it is copied to DTCM.
#target = #hal.executable.target<"coralnpu", "coralnpu-elf", {native_vector_size = 32 : i64}>
#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, Indirect>,
  #hal.pipeline.binding<storage_buffer, Indirect>
], flags = Indirect>
#translation = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<DoubleTilingExpert>, {enable_loop_peeling}>
#root = #iree_cpu.lowering_config<cache_parallel = [0, 0, 0], cache_reduction = [0, 0, 0], distribution = [0, 0, 0], promote_operands = [0, 1, 2], vector_common_parallel = [4, 16, 0], vector_reduction = [0, 0, 1]>
#elementwise = #iree_cpu.lowering_config<vector_common_parallel = [4, 16]>
// CHECK-LABEL: func.func @unbounded_tiles()
//  CHECK-NEXT:   %[[RHS:.+]] = bufferization.alloc_tensor() <{memory_space = #iree_codegen.workgroup_local}> : tensor<64x16xf32>
//   CHECK-NOT:   bufferization.alloc_tensor
//       CHECK:   %[[LHS_LOAD:.+]] = iree_codegen.load_from_buffer {{.*}} -> tensor<?x64xf32>
//       CHECK:   %[[RHS_LOAD:.+]] = iree_codegen.load_from_buffer {{.*}} -> tensor<64x16xf32>
//  CHECK-NEXT:   %[[RHS_LOCAL:.+]] = bufferization.materialize_in_destination %[[RHS_LOAD]] in %[[RHS]]
//       CHECK:   %[[MATMUL:.+]] = linalg.matmul {{.*}} ins(%[[LHS_LOAD]], %[[RHS_LOCAL]] : tensor<?x64xf32>, tensor<64x16xf32>)
//       CHECK:   %[[NEG:.+]] = linalg.generic {{.*}} ins(%[[MATMUL]] : tensor<?x16xf32>)
//       CHECK:   iree_codegen.store_to_buffer %[[MATMUL]]
//  CHECK-NEXT:   iree_codegen.store_to_buffer %[[NEG]]
hal.executable private @unbounded_tiles {
  hal.executable.variant public @coralnpu target(#target) {
    builtin.module {
      func.func @unbounded_tiles() attributes {translation_info = #translation} {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %m_i32 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
        %m = arith.index_castui %m_i32 : i32 to index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<?x64xf32, #hal.descriptor_type<storage_buffer>>{%m}
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<64x16xf32, #hal.descriptor_type<storage_buffer>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(Indirect) : memref<?x16xf32, #hal.descriptor_type<storage_buffer>>{%m}
        %b3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) alignment(64) offset(%c0) flags(Indirect) : memref<?x16xf32, #hal.descriptor_type<storage_buffer>>{%m}
        // expected-remark @+1 {{the tile has no static size bound; it stays in memory}}
        %3 = iree_codegen.load_from_buffer %0 : memref<?x64xf32, #hal.descriptor_type<storage_buffer>> -> tensor<?x64xf32>
        %4 = iree_codegen.load_from_buffer %1 : memref<64x16xf32, #hal.descriptor_type<storage_buffer>> -> tensor<64x16xf32>
        %5 = tensor.empty(%m) : tensor<?x16xf32>
        %6 = linalg.fill {lowering_config = #elementwise} ins(%cst : f32) outs(%5 : tensor<?x16xf32>) -> tensor<?x16xf32>
        // expected-remark @+1 {{the accumulator has no static size bound; it stays in memory}}
        %7 = linalg.matmul {lowering_config = #root} ins(%3, %4 : tensor<?x64xf32>, tensor<64x16xf32>) outs(%6 : tensor<?x16xf32>) -> tensor<?x16xf32>
        %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%7 : tensor<?x16xf32>) outs(%5 : tensor<?x16xf32>) attrs = {lowering_config = #elementwise} {
        ^bb0(%in: f32, %o: f32):
          %neg = arith.negf %in : f32
          linalg.yield %neg : f32
        } -> tensor<?x16xf32>
        iree_codegen.store_to_buffer %7, %2 : tensor<?x16xf32> into memref<?x16xf32, #hal.descriptor_type<storage_buffer>>
        // expected-remark @+1 {{the stored result does not come from a fused linalg op with static bounds; it is not staged in DTCM}}
        iree_codegen.store_to_buffer %8, %b3 : tensor<?x16xf32> into memref<?x16xf32, #hal.descriptor_type<storage_buffer>>
        return
      }
    }
  }
}

// -----

// A root op with more than one result: nothing is promoted, with a missed
// remark.
#target = #hal.executable.target<"coralnpu", "coralnpu-elf", {native_vector_size = 32 : i64}>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, Indirect>,
  #hal.pipeline.binding<storage_buffer, Indirect>
], flags = Indirect>
#translation = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<DoubleTilingExpert>, {enable_loop_peeling}>
#root = #iree_cpu.lowering_config<cache_parallel = [0, 0], cache_reduction = [0, 0], distribution = [0, 0], promote_operands = [0, 1], vector_common_parallel = [4, 16], vector_reduction = [0, 0]>
// CHECK-LABEL: func.func @multi_result_root()
//   CHECK-NOT:   bufferization.alloc_tensor
//   CHECK-NOT:   bufferization.materialize_in_destination
//       CHECK:   return
hal.executable private @multi_result_root {
  hal.executable.variant public @coralnpu target(#target) {
    builtin.module {
      func.func @multi_result_root() attributes {translation_info = #translation} {
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<4x16xf32, #hal.descriptor_type<storage_buffer>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(Indirect) : memref<4x16xf32, #hal.descriptor_type<storage_buffer>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(Indirect) : memref<4x16xf32, #hal.descriptor_type<storage_buffer>>
        %3 = iree_codegen.load_from_buffer %0 : memref<4x16xf32, #hal.descriptor_type<storage_buffer>> -> tensor<4x16xf32>
        %4 = tensor.empty() : tensor<4x16xf32>
        // expected-remark @+1 {{the root op has more than one result; the operands stay in memory}}
        %5:2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3 : tensor<4x16xf32>) outs(%4, %4 : tensor<4x16xf32>, tensor<4x16xf32>) attrs = {lowering_config = #root} {
        ^bb0(%in: f32, %o0: f32, %o1: f32):
          %neg = arith.negf %in : f32
          linalg.yield %in, %neg : f32, f32
        } -> (tensor<4x16xf32>, tensor<4x16xf32>)
        iree_codegen.store_to_buffer %5#0, %1 : tensor<4x16xf32> into memref<4x16xf32, #hal.descriptor_type<storage_buffer>>
        iree_codegen.store_to_buffer %5#1, %2 : tensor<4x16xf32> into memref<4x16xf32, #hal.descriptor_type<storage_buffer>>
        return
      }
    }
  }
}

// -----

// A root op without promote_operands: nothing changes.
#target = #hal.executable.target<"coralnpu", "coralnpu-elf", {native_vector_size = 32 : i64}>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, Indirect>
], flags = Indirect>
#translation = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<DoubleTilingExpert>, {enable_loop_peeling}>
#root = #iree_cpu.lowering_config<cache_parallel = [0, 0, 0], cache_reduction = [0, 0, 0], distribution = [0, 0, 0], vector_common_parallel = [4, 16, 0], vector_reduction = [0, 0, 1]>
#elementwise = #iree_cpu.lowering_config<vector_common_parallel = [4, 16]>
// CHECK-LABEL: func.func @no_promotion()
//   CHECK-NOT:   bufferization.alloc_tensor
//   CHECK-NOT:   bufferization.materialize_in_destination
//       CHECK:   return
hal.executable private @no_promotion {
  hal.executable.variant public @coralnpu target(#target) {
    builtin.module {
      func.func @no_promotion() attributes {translation_info = #translation} {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<64x128xf32, #hal.descriptor_type<storage_buffer>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<128x8xf32, #hal.descriptor_type<storage_buffer>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(Indirect) : memref<64x8xf32, #hal.descriptor_type<storage_buffer>>
        %3 = iree_codegen.load_from_buffer %0 : memref<64x128xf32, #hal.descriptor_type<storage_buffer>> -> tensor<64x128xf32>
        %4 = iree_codegen.load_from_buffer %1 : memref<128x8xf32, #hal.descriptor_type<storage_buffer>> -> tensor<128x8xf32>
        %5 = tensor.empty() : tensor<64x8xf32>
        %6 = linalg.fill {lowering_config = #elementwise} ins(%cst : f32) outs(%5 : tensor<64x8xf32>) -> tensor<64x8xf32>
        %7 = linalg.matmul {lowering_config = #root} ins(%3, %4 : tensor<64x128xf32>, tensor<128x8xf32>) outs(%6 : tensor<64x8xf32>) -> tensor<64x8xf32>
        iree_codegen.store_to_buffer %7, %2 : tensor<64x8xf32> into memref<64x8xf32, #hal.descriptor_type<storage_buffer>>
        return
      }
    }
  }
}
