// RUN: %iree_compile --compile-from=executable-sources \
// RUN:   --compile-to=executable-configurations %s | FileCheck %s

// The DTCM footprint model (CoralNPUTileSizeSelectionFootprint) on dispatches
// that the default IREE flow does not form from the input IR, so the inputs
// are executable sources. The target has no Zvt. tile_size_selection_dtcm.mlir
// has the other cases and the rules of the DTCM tile search; the DTCM tile
// budget here is 24 KB (the defaults).

// A fused producer whose result the dispatch also stores: the relu of A is the
// LHS of the matmul, and the dispatch stores it too. (The default IREE flow
// does not form this dispatch: it does not fuse a producer of a matmul input.)
// The footprint counts an output staging buffer for the stored relu tile, as
// for each result that leaves the fused ops. Untiled: A 64x64 f32 (16 KB) + B
// 64x32 f32 (8 KB) + relu 64x64 f32 (16 KB) + C 64x32 f32 (8 KB) = 48 KB. With
// 32 rows: A (8 KB) + B (8 KB) + relu (8 KB) + C (4 KB) = 28 KB, which does not
// fit. With 16 rows: A (4 KB) + B (8 KB) + relu (4 KB) + C (2 KB) = 18 KB.
// CHECK-LABEL: func.func @stored_producer()
//       CHECK:   linalg.matmul
//  CHECK-SAME:     cache_parallel = [16, 0, 0], cache_reduction = [0, 0, 0], distribution = [16, 0, 0]
#executable_target = #hal.executable.target<"coralnpu", "embedded-elf-riscv_32", {cpu = "", cpu_features = "+m,+f,+zvl128b,+zve32f", data_layout = "e-m:e-p:32:32-i64:64-n32-S128", debug_symbols = false, iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, max_stack_allocation_size = 3072 : i64, native_vector_size = 32 : i64, target_abi = "ilp32", target_triple = "riscv32-unknown-unknown-eabi-elf"}>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, Indirect>,
  #hal.pipeline.binding<storage_buffer, Indirect>
], flags = Indirect>
hal.executable private @stored_producer {
  hal.executable.variant public @embedded_elf_riscv_32 target(#executable_target) {
    hal.executable.export public @stored_producer ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @stored_producer() {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x64xf32>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x32xf32>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64x64xf32>>
        %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64x32xf32>>
        %4 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [64, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x64xf32>> -> tensor<64x64xf32>
        %5 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [64, 32], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x32xf32>> -> tensor<64x32xf32>
        %6 = tensor.empty() : tensor<64x64xf32>
        %relu = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%4 : tensor<64x64xf32>) outs(%6 : tensor<64x64xf32>) {
        ^bb0(%in: f32, %out: f32):
          %v = arith.maximumf %in, %cst : f32
          linalg.yield %v : f32
        } -> tensor<64x64xf32>
        iree_tensor_ext.dispatch.tensor.store %relu, %2, offsets = [0, 0], sizes = [64, 64], strides = [1, 1] : tensor<64x64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64x64xf32>>
        %7 = tensor.empty() : tensor<64x32xf32>
        %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<64x32xf32>) -> tensor<64x32xf32>
        %9 = linalg.matmul ins(%relu, %5 : tensor<64x64xf32>, tensor<64x32xf32>) outs(%8 : tensor<64x32xf32>) -> tensor<64x32xf32>
        iree_tensor_ext.dispatch.tensor.store %9, %3, offsets = [0, 0], sizes = [64, 32], strides = [1, 1] : tensor<64x32xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64x32xf32>>
        return
      }
    }
  }
}

// Multi-op reduction with a stored 2D parallel producer (%mul = %x * %w), a
// non-root %max reduction over %mul, an unfused 2D parallel producer (%ex) that
// only feeds the root %sum reduction, and a 2D %div consumer. Storing %mul
// prevents RematerializeParallelOpsPass from folding it into %max so it gets an
// inner-parallel lowering_config, while %ex (kept unfused by an external
// capture) only feeds %sum so syncNonRootRVVConfigs leaves %ex without a
// lowering_config to fuse into %sum.
// Footprint per row: x (2 KB) + w (2 KB) + stored mul (2 KB) + div (2 KB) =
// 8 KB/row (128 KB untiled). With 24 KB, 2 rows (16 KB) fit.
// CHECK-LABEL: func.func @softmax_scaled_max()
//       CHECK:   linalg.generic
//  CHECK-SAME:     iterator_types = ["parallel", "parallel"]
//  CHECK-SAME:     lowering_config = #iree_cpu.lowering_config<vector_common_parallel = [1, 0], vector_inner_parallel = [0, 16]>
//       CHECK:   linalg.generic
//  CHECK-SAME:     iterator_types = ["parallel", "reduction"]
//  CHECK-SAME:     lowering_config = #iree_cpu.lowering_config<vector_common_parallel = [1, 0], vector_reduction = [0, 16]>
//       CHECK:   linalg.generic
//  CHECK-SAME:     iterator_types = ["parallel", "parallel"]
//   CHECK-NOT:     lowering_config
//       CHECK:   linalg.generic
//  CHECK-SAME:     iterator_types = ["parallel", "reduction"]
//  CHECK-SAME:     lowering_config = #iree_cpu.lowering_config<cache_parallel = [2, 0], cache_reduction = [0, 0], distribution = [2, 0], vector_common_parallel = [1, 0], vector_reduction = [0, 16]>
//       CHECK:   linalg.generic
//  CHECK-SAME:     iterator_types = ["parallel", "parallel"]
//  CHECK-SAME:     lowering_config = #iree_cpu.lowering_config<vector_common_parallel = [1, 0], vector_inner_parallel = [0, 16]>
hal.executable private @softmax_scaled_max {
  hal.executable.variant public @embedded_elf_riscv_32 target(#executable_target) {
    hal.executable.export public @softmax_scaled_max ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @softmax_scaled_max() {
        %c0 = arith.constant 0 : index
        %cst0 = arith.constant 0.000000e+00 : f32
        %ninf = arith.constant 0xFF800000 : f32
        %b0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x512xf32>>
        %b1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x512xf32>>
        %b2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16x512xf32>>
        %b3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16x512xf32>>
        %x = iree_tensor_ext.dispatch.tensor.load %b0, offsets = [0, 0], sizes = [16, 512], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x512xf32>> -> tensor<16x512xf32>
        %w = iree_tensor_ext.dispatch.tensor.load %b1, offsets = [0, 0], sizes = [16, 512], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x512xf32>> -> tensor<16x512xf32>
        %e2 = tensor.empty() : tensor<16x512xf32>
        %e1 = tensor.empty() : tensor<16xf32>
        %mul = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%x, %w : tensor<16x512xf32>, tensor<16x512xf32>) outs(%e2 : tensor<16x512xf32>) {
        ^bb0(%in0: f32, %in1: f32, %out: f32):
          %p = arith.mulf %in0, %in1 : f32
          linalg.yield %p : f32
        } -> tensor<16x512xf32>
        iree_tensor_ext.dispatch.tensor.store %mul, %b2, offsets = [0, 0], sizes = [16, 512], strides = [1, 1] : tensor<16x512xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16x512xf32>>
        %fmax = linalg.fill ins(%ninf : f32) outs(%e1 : tensor<16xf32>) -> tensor<16xf32>
        %max = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%mul : tensor<16x512xf32>) outs(%fmax : tensor<16xf32>) {
        ^bb0(%in: f32, %acc: f32):
          %m = arith.maxnumf %in, %acc : f32
          linalg.yield %m : f32
        } -> tensor<16xf32>
        %ex = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%x, %max : tensor<16x512xf32>, tensor<16xf32>) outs(%e2 : tensor<16x512xf32>) {
        ^bb0(%in: f32, %m: f32, %out: f32):
          %idx = tensor.extract %max[%c0] : tensor<16xf32>
          %d = arith.subf %in, %m : f32
          %d2 = arith.addf %d, %idx : f32
          %e = math.exp %d2 : f32
          linalg.yield %e : f32
        } -> tensor<16x512xf32>
        %fsum = linalg.fill ins(%cst0 : f32) outs(%e1 : tensor<16xf32>) -> tensor<16xf32>
        %sum = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%ex : tensor<16x512xf32>) outs(%fsum : tensor<16xf32>) {
        ^bb0(%in: f32, %acc: f32):
          %s = arith.addf %in, %acc : f32
          linalg.yield %s : f32
        } -> tensor<16xf32>
        %div = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%x, %max, %sum : tensor<16x512xf32>, tensor<16xf32>, tensor<16xf32>) outs(%e2 : tensor<16x512xf32>) {
        ^bb0(%in: f32, %m: f32, %s: f32, %out: f32):
          %d = arith.subf %in, %m : f32
          %e = math.exp %d : f32
          %q = arith.divf %e, %s : f32
          linalg.yield %q : f32
        } -> tensor<16x512xf32>
        iree_tensor_ext.dispatch.tensor.store %div, %b3, offsets = [0, 0], sizes = [16, 512], strides = [1, 1] : tensor<16x512xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16x512xf32>>
        return
      }
    }
  }
}
