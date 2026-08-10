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

#ifndef COMPILER_TRANSFORMS_CORALNPUTILESIZESELECTIONANALYSIS_H_
#define COMPILER_TRANSFORMS_CORALNPUTILESIZESELECTIONANALYSIS_H_

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/TypeID.h"

namespace mlir::coralnpu_compiler {

struct CoralNPUTileSizeSelectionAnalysis {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      CoralNPUTileSizeSelectionAnalysis)

  LogicalResult status = failure();

  TilingInterface rootTilingOp = nullptr;
  // TODO: handle sub-byte elements
  int64_t elemSizeBytes = 0;
  SmallVector<int64_t> staticLoopRanges;

  // parallelLoops and reductionLoops are ordered inner most first, e.g.,
  // parallelLoops[0] is the inner most parallel loop.
  SmallVector<size_t> parallelLoops;
  SmallVector<size_t> reductionLoops;

  explicit CoralNPUTileSizeSelectionAnalysis(Operation *op);
};

}  // namespace mlir::coralnpu_compiler

#endif  // COMPILER_TRANSFORMS_CORALNPUTILESIZESELECTIONANALYSIS_H_
