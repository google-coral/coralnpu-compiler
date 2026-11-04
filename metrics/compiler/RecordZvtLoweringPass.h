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

// Records whether the matrix codegen lowered each dispatch to the Zvt matrix
// unit, for the static Zvt lowering status of the metrics bundle.

#ifndef METRICS_COMPILER_RECORDZVTLOWERINGPASS_H_
#define METRICS_COMPILER_RECORDZVTLOWERINGPASS_H_

// MLIR:
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

// Standard library:
#include <memory>

namespace mlir::coralnpu_compiler {

// Returns a pass that completes the `zvt` entry of the `coralnpu.metrics`
// attribute of each dispatch function (MetricsAttr.h), which
// RecordTileGridPass starts with `candidate`, with `lowered`: whether the
// function has a Zvt matrix multiply. When `lowered` is false, it removes
// `zvt_grid` from `tile_grid`. A function without the entry gets one only if
// it has a Zvt matrix multiply. Runs right after the matrix codegen, on Zvt
// targets.
std::unique_ptr<InterfacePass<FunctionOpInterface>>
createRecordZvtLoweringPass();

// Registers `RecordZvtLoweringPass` with the MLIR pass registry.
void registerRecordZvtLoweringPass();

}  // namespace mlir::coralnpu_compiler

#endif  // METRICS_COMPILER_RECORDZVTLOWERINGPASS_H_
