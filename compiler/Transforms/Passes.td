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

def CoralNPUAffinityAnnotation : Pass<"coralnpu-affinity-annotation", "mlir::ModuleOp"> {
  let summary = "Annotates operations with CoralNPU device affinity";

  let description = [{
    Annotates operations that should be placed on the CoralNPU with device affinity attribute.
  }];

  let constructor = [{
    ::mlir::coralnpu_compiler::createCoralNPUAffinityAnnotationPass()
  }];

  let options = [
    Option<"ioMinThresholdBytes", "io-min-threshold-bytes", "int64_t",
           /*default=*/"0",
           "Minimum estimated dispatch input/output bytes required before "
           "placing an operation on CoralNPU. Use 0 for bring-up tests.">,
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

def CoralNPUTileSizeSelectionRegister : InterfacePass<"coralnpu-tile-size-selection-register", "mlir::FunctionOpInterface"> {
  let summary = "Selects register-level tile sizes for CoralNPU operations";
  let description = [{
    This pass determines the optimal register-level tile sizes for CoralNPU.
    It resolves target-specific register alignments (vector width, unroll factors)
    and reduction alignments to ensure efficient vectorization and unrolling.
  }];
  let constructor = [{
    ::mlir::coralnpu_compiler::createCoralNPUTileSizeSelectionRegisterPass()
  }];
  let options = [
    Option<"numVectorRegisters", "num-vector-registers", "int",
           /*default=*/"32",
           "Number of vector registers.">,
    Option<"vectorRegisterWidthBits", "vector-register-width-bits", "int",
           /*default=*/"0",
           "Vector register width in bits (0 for auto-detect from target).">,
    Option<"vectorAlignment", "vector-alignment", "int",
           /*default=*/"0",
           "Alignment for vector parallel loops (0 for auto).">,
    Option<"unrollAlignment", "unroll-alignment", "int",
           /*default=*/"0",
           "Alignment for unrolled parallel loops (0 for auto).">,
    Option<"reductionAlignment", "reduction-alignment", "int",
           /*default=*/"1",
           "Alignment for reduction loops.">,
    Option<"parallelAlignment", "parallel-alignment", "int",
           /*default=*/"0",
           "Alignment for generic parallel loops (0 for auto).">,
  ];
}

def CoralNPUTileSizeSelectionDTCM : InterfacePass<"coralnpu-tile-size-selection-dtcm", "mlir::FunctionOpInterface"> {
  let summary = "Selects DTCM-level tile sizes for CoralNPU operations";
  let description = [{
    This pass determines the optimal DTCM-level tile sizes for CoralNPU.
    It uses a footprint estimation model to iteratively shrink the tile sizes
    until the estimated working memory footprint fits within the DTCM budget,
    using register tile sizes as alignment constraints.
  }];
  let constructor = [{
    ::mlir::coralnpu_compiler::createCoralNPUTileSizeSelectionDTCMPass()
  }];
  let options = [
    Option<"dtcmSizeKb", "dtcm-size-kb", "int",
           /*default=*/"8",
           "Size of the DTCM in KB.">,
  ];
}

def CoralNPUTileSizeSelectionWorkgroup : InterfacePass<"coralnpu-tile-size-selection-workgroup", "mlir::FunctionOpInterface"> {
  let summary = "Selects workgroup-level tile sizes for CoralNPU operations";
  let description = [{
    This pass determines the workgroup-level tile sizes for CoralNPU,
    typically matching the DTCM parallel tile sizes.
  }];
  let constructor = [{
    ::mlir::coralnpu_compiler::createCoralNPUTileSizeSelectionWorkgroupPass()
  }];
}

#endif // IREE_CORALNPU_PASSES

