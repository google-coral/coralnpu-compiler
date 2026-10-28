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
//
//===----------------------------------------------------------------------===//
// CoralNPU RVV Vector Tile Size and Pipeline Selection
//===----------------------------------------------------------------------===//

#ifndef COMPILER_TRANSFORMS_CORALNPUTILESIZESELECTIONRVV_H_
#define COMPILER_TRANSFORMS_CORALNPUTILESIZESELECTIONRVV_H_

#include "compiler/Transforms/CoralNPUTileSizeSelectionUtils.h"

// IREE:
#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUTypes.h"

// MLIR:
#include "mlir/Interfaces/FunctionInterfaces.h"

// LLVM:
#include "llvm/ADT/ArrayRef.h"

// Standard library:
#include <optional>

namespace mlir::coralnpu_compiler {

// Refines the RVV vector tile sizes of a non-Zvt root op (clamping the shared
// reduction tile to the smallest tile of `root.nonRootOps`), returning the CPU
// pipeline, or `std::nullopt` to keep IREE's strategy sizes.
std::optional<iree_compiler::IREE::CPU::LoweringPipeline> refineRVVTileSizes(
    FunctionOpInterface funcOp, const DTCMRoot &root,
    MutableArrayRef<int64_t> vectorParallelSizes,
    MutableArrayRef<int64_t> vectorReductionSizes);

// Writes synchronized vector lowering_configs for `root.nonRootOps` (except
// parallel producers that only feed reductions) after DTCM clamping.
void syncNonRootRVVConfigs(const DTCMRoot &root,
                           ArrayRef<int64_t> vectorParallelSizes,
                           ArrayRef<int64_t> vectorReductionSizes);

}  // namespace mlir::coralnpu_compiler

#endif  // COMPILER_TRANSFORMS_CORALNPUTILESIZESELECTIONRVV_H_
