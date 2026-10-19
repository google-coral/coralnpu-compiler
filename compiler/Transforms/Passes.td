// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

include "mlir/Pass/PassBase.td"

#ifndef IREE_CORALNPU_PASSES
#define IREE_CORALNPU_PASSES

def CoralNPUMaterializeDeviceTopology : Pass<"coralnpu-materialize-device-topology", "mlir::ModuleOp"> {
  let summary = "Materializes the CoralNPU device topology on the module";

  let description = [{
    Configures stream.topology and stream.affinity.default on the module when
    targeting both a host device and a CoralNPU device.
  }];

  let constructor = [{
    ::mlir::coralnpu_compiler::createCoralNPUMaterializeDeviceTopologyPass()
  }];
}

def CoralNPUAffinityAnnotation : Pass<"coralnpu-affinity-annotation", "mlir::ModuleOp"> {
  let summary = "Assigns dispatches to CoralNPU or the host";

  let description = [{
    Places supported dispatches on CoralNPU or the host using a roofline
    latency model, and unsupported ones on the host.
  }];

  let constructor = [{
    ::mlir::coralnpu_compiler::createCoralNPUAffinityAnnotationPass()
  }];

  let options = [
    Option<"rooflineSpeedupThreshold", "roofline-speedup-threshold", "double",
           /*default=*/"1.0",
           "Minimum estimated speedup (CPU latency / NPU latency) required to "
           "place a dispatch on CoralNPU; 0 places all supported dispatches "
           "(default: 1.0).">,
  ];
}

def CoralNPUDumpAffinityExecutionProfile : Pass<"coralnpu-dump-affinity-execution-profile", "mlir::ModuleOp"> {
  let summary = "Dumps execution profile statistics grouped by affinity";

  let description = [{
    Walks the stream operations and dumps execution profile statistics (dispatches, data sizes, work estimates) grouped by affinity.
  }];

  let constructor = [{
    ::mlir::coralnpu_compiler::createCoralNPUDumpAffinityExecutionProfilePass()
  }];

  let options = [
    Option<"outputFormat", "output-format", "::mlir::coralnpu_compiler::DumpOutputFormat",
           /*default=*/"::mlir::coralnpu_compiler::DumpOutputFormat::Pretty",
           "Format of the output statistics.",
           [{::llvm::cl::values(
             clEnumValN(::mlir::coralnpu_compiler::DumpOutputFormat::Pretty, "pretty", "Human-readable pretty printed output."),
             clEnumValN(::mlir::coralnpu_compiler::DumpOutputFormat::Verbose, "verbose", "Pretty printed output with additional IR."),
             clEnumValN(::mlir::coralnpu_compiler::DumpOutputFormat::CSV, "csv", "Comma separated values."),
             clEnumValN(::mlir::coralnpu_compiler::DumpOutputFormat::JSON, "json", "JSON format.")
           )}]>,
    Option<"outputFile", "output-file", "std::string",
           /*default=*/"\"\"",
           "File path to write to; or `` for stderr or `-` for stdout.">
  ];
}

def CoralNPUTileSizeSelectionDTCM : InterfacePass<"coralnpu-tile-size-selection-dtcm", "mlir::FunctionOpInterface"> {
  let summary = "Selects DTCM-level tile sizes for CoralNPU operations";
  let description = [{
    This pass determines the optimal DTCM-level tile sizes for CoralNPU. It runs
    after LLVMCPUSelectLoweringStrategyPass and refines the lowering config of
    the root op: it sets the cache_parallel and cache_reduction tile sizes.
    It uses a footprint estimation model to iteratively shrink the tile sizes
    until the estimated DTCM footprint of one tile fits within the DTCM tile
    budget, using the vector tile sizes of the strategy pass as alignment
    constraints and preferring exact divisors of the loop ranges. The
    footprint is the sum of the DTCM buffers of the root op and of the ops
    fused with it (input slices, output staging buffers, an accumulator for a
    split reduction, and the scratch tiles of the Zvt asm). The budget is the
    DTCM size minus the stack and the reserve, rounded down to 4 KB.
    For Zvt matrix contractions and convolutions it sets M/N vector tiles of
    up to 64 and vector reduction tiles of 1, and enables loop peeling;
    otherwise the RVV sizes of the strategy pass are kept. The reduction loops
    of Zvt ops and of convolutions are not split: they stream one step at a
    time.
  }];
  let constructor = [{
    ::mlir::coralnpu_compiler::createCoralNPUTileSizeSelectionDTCMPass()
  }];
  let options = [
    Option<"dtcmSizeKb", "dtcm-size-kb", "int",
           /*default=*/"32",
           "Size of the DTCM in KB.">,
    Option<"stackSizeKb", "stack-size-kb", "int",
           /*default=*/"4",
           "Size of the stack in DTCM, in KB.">,
    Option<"dtcmReserveKb", "dtcm-reserve-kb", "int",
           /*default=*/"4",
           "DTCM space in KB to keep free for static data.">,
    Option<"buffersPerOperand", "buffers-per-operand", "int",
           /*default=*/"1",
           "Number of DTCM buffers for each input slice (2 for double "
           "buffering).">
  ];
}

def CoralNPUTileSizeSelectionWorkgroup : InterfacePass<"coralnpu-tile-size-selection-workgroup", "mlir::FunctionOpInterface"> {
  let summary = "Selects workgroup-level tile sizes for CoralNPU operations";
  let description = [{
    This pass determines the workgroup-level tile sizes for CoralNPU. It runs
    after CoralNPUTileSizeSelectionDTCM and sets the distribution tile sizes
    of the root op to the DTCM parallel tile sizes (cache_parallel).
  }];
  let constructor = [{
    ::mlir::coralnpu_compiler::createCoralNPUTileSizeSelectionWorkgroupPass()
  }];
}

def CoralNPUPromoteOperands : InterfacePass<"coralnpu-promote-operands", "mlir::FunctionOpInterface"> {
  let summary = "Copies the tiles of the operands of a root op to DTCM";
  let description = [{
    Runs after the cache-level tiling, on a root op whose lowering config has
    a `promote_operands` entry (set by CoralNPUTileSizeSelectionDTCM). In the
    workgroup tile (DTCM tile), it copies each binding tile that the root op
    and the ops fused with it read into a DTCM tensor, with a
    `bufferization.materialize_in_destination` op. The root op accumulates in
    DTCM if the tile splits K or if its result is stored, and each stored
    result goes through an output staging buffer in DTCM: the store op copies
    it to memory when it bufferizes. After bufferization,
    CoralNPULowerDTCMCopies lowers the copies.

    First, the pass clones into the tile loop the elementwise producers (e.g.
    dequantizations) that IREE left out of it, because no tiled loop indexes
    their result. Thus their binding inputs get DTCM copies like the inputs of
    the producers that IREE fuses (the footprint model of
    CoralNPUTileSizeSelectionDTCM counts them too).

    The DTCM buffers are static `bufferization.alloc_tensor` ops in
    `#iree_codegen.workgroup_local`, with the upper bounds of the tile sizes,
    at the start of the workgroup loop body (or of the function if there is
    no workgroup loop). Each tile with dynamic sizes uses a slice of its
    buffer. Thus loop peeling does not clone the buffers, and the buffers
    become `memref.alloc` ops in the entry block of the function.
  }];
  let constructor = [{
    ::mlir::coralnpu_compiler::createCoralNPUPromoteOperandsPass()
  }];
  let dependentDialects = [
    "::mlir::bufferization::BufferizationDialect",
    "::mlir::linalg::LinalgDialect",
    "::mlir::tensor::TensorDialect",
  ];
}

def CoralNPULowerDTCMCopies : InterfacePass<"coralnpu-lower-dtcm-copies", "mlir::FunctionOpInterface"> {
  let summary = "Lowers the copies between memory and DTCM to vector loads and stores";
  let description = [{
    Runs after bufferization and the vector lowering (in the pre-LLVM hook of
    the CoralNPU target). It lowers each copy (a `memref.copy`, or a linalg
    copy on memrefs) whose source or target is in DTCM
    (`#iree_codegen.workgroup_local`) and whose rows are contiguous to loops
    of vector loads and stores along the rows. The last vector of a row has a
    mask if the length of the rows is not a multiple of the vector size.
    These copies are the tile copies of CoralNPUPromoteOperands; without this
    pass, the pre-LLVM hook makes a copy of fewer than 1024 elements one
    vector (GenericVectorizationPass), and IREE lowers the other copies to
    scalar loops.
    TODO(DMA): lower the copies to DMA transfers here, and keep the vector
    loops as the fallback.
  }];
  let constructor = [{
    ::mlir::coralnpu_compiler::createCoralNPULowerDTCMCopiesPass()
  }];
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::vector::VectorDialect",
  ];
}

def CoralNPULimitLoopUnrolling : Pass<"coralnpu-limit-loop-unrolling", "mlir::func::FuncOp"> {
  let summary = "Limit loop unrolling by annotating loops";
  let description = [{
    This pass limits unrolling with llvm.loop_annotation to disable further unrolling in LLVM.
  }];
  let constructor = [{
    ::mlir::coralnpu_compiler::createCoralNPULimitLoopUnrollingPass()
  }];
  let options = [
    Option<"maxLoopUnrolling", "max-loop-unrolling", "int", /*default=*/"32",
           "Maximum unroll factor allowed for any loop (default: 32). LLVM unrolling can be too aggressive, which leads to ITCM overflow. Setting this to 1 effectively disables unrolling.">
  ];
}

def CoralNPUOptimizeVectorLoops : Pass<"coralnpu-optimize-vector-loops", "mlir::func::FuncOp"> {
  let summary = "Restructure vectorized loops";
  let description = [{
    Accumulates loop-carried vector.reduction <add> sums in vectors, unrolls
    and unroll-and-jams small vectorized loops within the vector register
    budget, and interchanges 2-D broadcast nests.
  }];
  let constructor = [{
    ::mlir::coralnpu_compiler::createCoralNPUOptimizeVectorLoopsPass()
  }];
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::vector::VectorDialect"
  ];
  let options = [
    Option<"maxLoopUnrolling", "max-loop-unrolling", "int", /*default=*/"32",
           "Maximum unroll factor allowed for any loop in each unrolling step (default: 32).">,
    Option<"numVectorRegisters", "num-vector-registers", "int", /*default=*/"32",
           "Number of vector registers (default: 32).">,
    Option<"fastMath", "fast-math", "bool", /*default=*/"true",
           "Without the reassoc flag, also accumulate float sums of products of values widened from <= 16-bit types in vectors, which reorders the sum (default: true).">
  ];
}

def CoralNPUApproximateMath : Pass<"coralnpu-approximate-math", "mlir::func::FuncOp"> {
  let summary = "Replace math ops with fast approximations";
  let description = [{
    Replaces f32, f16 and bf16 math.tanh, math.exp and math.erf, and the
    expansion of chlo.erfc that StableHLO emits, with approximations that are
    cheaper than the polynomial approximations from MLIR: a clamped [7/6]
    rational function for tanh, a range-reduced degree-6 polynomial for exp
    and Abramowitz & Stegun 7.1.28 for erf. They have no selects or table
    lookups. Inputs are clamped, so NaN inputs do not produce NaN results.
  }];
  let constructor = [{
    ::mlir::coralnpu_compiler::createCoralNPUApproximateMathPass()
  }];
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::math::MathDialect"
  ];
}

def CoralNPURestoreConvInputChannel : InterfacePass<"coralnpu-restore-conv-input-channel", "mlir::FunctionOpInterface"> {
  let summary = "Restore the unit input channel of convolutions";
  let description = [{
    IREE's unit dim folding also drops the unit input channel of a
    convolution with a single-channel input, e.g. the first convolution of an
    audio or grayscale image model. The result has an output channel but no
    input channel, which IREE's convolution handling does not expect, so it
    is not decomposed and vectorized like other convolutions. This pass
    restores the unit input channel of channel-last convolutions, to run
    before InsertBatchDimForBatchlessConv.
  }];
  let constructor = [{
    ::mlir::coralnpu_compiler::createCoralNPURestoreConvInputChannelPass()
  }];
  let dependentDialects = ["::mlir::tensor::TensorDialect"];
}

def CoralNPUPromoteBF16Accumulator : Pass<"coralnpu-promote-bf16-accumulator"> {
  let summary = "Promote BF16 contraction, convolution, and reduction accumulators to FP32";
  let description = [{
    Rewrites BF16 contractions, convolutions, and sum/product reductions to
    accumulate into an FP32 init and truncate once after the reduction.
    Scheduled in global preprocessing so host and device agree.
    Host-only modules keep dynamic shapes in BF16.
  }];
  let constructor = [{
    ::mlir::coralnpu_compiler::createCoralNPUPromoteBF16AccumulatorPass()
  }];
}

def CoralNPURaiseApplyScale : Pass<"coralnpu-raise-apply-scale"> {
  let summary = "Raise the 64-bit expansion of tosa.apply_scale back to the op";
  let description = [{
    IREE's TOSA input conversion lowers tosa.apply_scale to 64-bit integer
    arithmetic, which targets without 64-bit vector elements, such as
    CoralNPU (Zve32x/Zve32f), scalarize. This pass recovers tosa.apply_scale
    from that arithmetic (and folds identity scf.if dynamic-broadcast
    branches), so that codegen lowers it with 32-bit operations on those
    targets. Scheduled in global preprocessing, before GlobalOptimization's
    integer optimizations and the affinity annotation; modules that target
    only other devices are left as they are.
  }];
  let constructor = [{
    ::mlir::coralnpu_compiler::createCoralNPURaiseApplyScalePass()
  }];
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::tosa::TosaDialect"
  ];
}

def CoralNPUMatrixCodegen : InterfacePass<"coralnpu-matrix-codegen", "mlir::FunctionOpInterface"> {
  let summary = "Lowers vector.contract to Zvt matrix inline assembly";
  let description = [{
    Lowers FP32, BF16 and INT8 vector.contract ops with a 16x16, 16x32,
    32x16, 32x32, 16x64, 64x16 or partial (M < 16, N in {8, 12})
    accumulator (and the scf.for K loop that yields them, when
    fusable) to Zvt inline assembly holding the accumulators in
    mt0..mt12, and +zvfbfwma BF16 vecmat K loops (VLEN 128) to
    vfwmaccbf16.vf. Other contractions in the function get MLIR's generic
    contract lowering.
  }];
  let constructor = "::mlir::coralnpu_compiler::createCoralNPUMatrixCodegenPass()";
  let dependentDialects = [
    "::mlir::arith::ArithDialect", "::mlir::LLVM::LLVMDialect",
    "::mlir::memref::MemRefDialect", "::mlir::vector::VectorDialect",
    "::mlir::iree_compiler::IREE::Codegen::IREECodegenDialect"
  ];
}

#endif // IREE_CORALNPU_PASSES

