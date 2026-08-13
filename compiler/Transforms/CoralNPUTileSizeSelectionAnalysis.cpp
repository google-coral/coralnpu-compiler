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

#include "compiler/Transforms/CoralNPUTileSizeSelectionAnalysis.h"

#include <algorithm>

#include "iree/compiler/Codegen/Utils/CPUUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

using namespace mlir;
using namespace mlir::iree_compiler;

namespace mlir::coralnpu_compiler {

namespace {

// Classifies loops of a TilingInterface op into parallel and reduction loops.
void classifyLoops(TilingInterface tilingOp,
                   SmallVectorImpl<size_t> &parallelLoops,
                   SmallVectorImpl<size_t> &reductionLoops) {
  auto iterTypes = tilingOp.getLoopIteratorTypes();

  for (auto [index, iterType] : llvm::enumerate(iterTypes)) {
    switch (iterType) {
      case utils::IteratorType::parallel:
        parallelLoops.push_back(index);
        break;
      case utils::IteratorType::reduction:
        reductionLoops.push_back(index);
        break;
    }
  }
  std::reverse(parallelLoops.begin(), parallelLoops.end());
  std::reverse(reductionLoops.begin(), reductionLoops.end());
}

// Helper to get static loop ranges from TilingInterface
SmallVector<int64_t> getStaticLoopRanges(TilingInterface op) {
  OpBuilder builder(op);
  SmallVector<Range> loopRange = op.getIterationDomain(builder);
  return llvm::map_to_vector(loopRange, [](Range r) -> int64_t {
    std::optional<int64_t> intVal = getConstantIntValue(r.size);
    return intVal ? intVal.value() : ShapedType::kDynamic;
  });
}

}  // namespace

CoralNPUTileSizeSelectionAnalysis::CoralNPUTileSizeSelectionAnalysis(
    Operation *op) {
  auto funcOp = dyn_cast<FunctionOpInterface>(op);
  if (!funcOp) return;

  SmallVector<Operation *> computeOps = getComputeOps(funcOp);
  auto rootOpOr = getRootOperation(computeOps);
  if (failed(rootOpOr) || !rootOpOr.value()) return;

  rootTilingOp = dyn_cast<TilingInterface>(*rootOpOr);
  if (!rootTilingOp) {
    rootOpOr.value()->emitWarning(
        "root operation is not a TilingInterface; skipping tile size "
        "selection");
    return;
  }

  // Element size
  if (rootTilingOp->getNumResults() > 0) {
    if (auto type =
            dyn_cast<ShapedType>(rootTilingOp->getResult(0).getType())) {
      auto elemType = type.getElementType();
      if (!elemType.isIntOrFloat()) {
        rootTilingOp->emitWarning("only integer and float types are supported");
        return;
      }
      auto elemBitWidth = elemType.getIntOrFloatBitWidth();
      if (elemBitWidth < 8) {
        rootTilingOp->emitWarning(
            "sub-byte types (e.g. i1, i4) are not supported");
        return;
      }
      elemSizeBytes = elemBitWidth / 8;
    }
  }

  // Static loop ranges
  staticLoopRanges = getStaticLoopRanges(rootTilingOp);
  if (llvm::is_contained(staticLoopRanges, ShapedType::kDynamic)) {
    rootTilingOp->emitWarning("dynamic shapes are not supported");
    return;
  }

  // Loop classification
  classifyLoops(rootTilingOp, parallelLoops, reductionLoops);

  status = success();
}

}  // namespace mlir::coralnpu_compiler
