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

#include "metrics/compiler/RecordTileGridPass.h"

#include "compiler/Transforms/CoralNPUTileSizeSelectionUtils.h"
#include "metrics/compiler/MetricsAttr.h"

// IREE:
#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUTypes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Utils/CPUUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"

// MLIR:
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Pass/PassRegistry.h"

// LLVM:
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

// Standard library:
#include <algorithm>
#include <memory>
#include <utility>

namespace mlir::coralnpu_compiler {

namespace {

namespace CPU = iree_compiler::IREE::CPU;

// Returns the effective tile sizes of a tiling level made of the |levels| of
// |config| (e.g. its parallel and reduction loops): the positive sizes of the
// levels clamped to |outer| (the tile of the enclosing level), and |outer| for
// loops left untiled (`0`).
SmallVector<int64_t> getTileSizes(CPU::LoweringConfigAttr config,
                                  ArrayRef<CPU::TilingLevel> levels,
                                  Operation *op, ArrayRef<int64_t> outer) {
  SmallVector<int64_t> tile(outer);
  for (CPU::TilingLevel level : levels) {
    for (auto [index, size] : llvm::enumerate(config.getStaticTilingLevelSizes(
             llvm::to_underlying(level), op))) {
      if (index < tile.size() && size > 0) {
        tile[index] = std::min(tile[index], size);
      }
    }
  }
  return tile;
}

class RecordTileGridPass final
    : public PassWrapper<RecordTileGridPass,
                         InterfacePass<FunctionOpInterface>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RecordTileGridPass)

  StringRef getArgument() const override {
    return "coralnpu-metrics-record-tile-grid";
  }

  StringRef getDescription() const override {
    return "Records the tile grid of each dispatch in its coralnpu.metrics "
           "attribute";
  }

  // Records the tile grid of the root op of the dispatch function, if it has
  // one.
  // TODO(sflur): Add the DTCM estimate of the DTCM tile from
  // DTCMFootprintModel in
  // compiler/Transforms/CoralNPUTileSizeSelectionFootprint.h.
  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();
    SmallVector<Operation *> computeOps = iree_compiler::getComputeOps(funcOp);
    auto [rootOp, config] = getTiledRootOp(computeOps);
    // Dispatches without a tiled root op (e.g. copies) have no tile grid.
    if (!rootOp || !config) return;
    auto tilingOp = dyn_cast<TilingInterface>(rootOp);
    if (!tilingOp) return;
    // The tile sizes of dynamic loops are not known at compile time.
    std::optional<SmallVector<int64_t>> loops = getStaticLoopRanges(tilingOp);
    if (!loops || loops->empty()) return;

    SmallVector<int64_t> workgroup = getTileSizes(
        config, {CPU::TilingLevel::DistributionTiles}, rootOp, *loops);
    SmallVector<int64_t> dtcm =
        getTileSizes(config,
                     {CPU::TilingLevel::CacheParallelTiles,
                      CPU::TilingLevel::CacheReductionTiles},
                     rootOp, workgroup);
    SmallVector<int64_t> vector =
        getTileSizes(config,
                     {CPU::TilingLevel::VectorCommonParallelTiles,
                      CPU::TilingLevel::VectorReductionTiles,
                      CPU::TilingLevel::VectorInnerParallelTiles},
                     rootOp, dtcm);

    Builder builder(funcOp.getContext());
    NamedAttrList grid;
    grid.set("op", builder.getStringAttr(rootOp->getName().getStringRef()));
    grid.set("loops", builder.getI64ArrayAttr(*loops));
    grid.set("workgroup", builder.getI64ArrayAttr(workgroup));
    grid.set("dtcm", builder.getI64ArrayAttr(dtcm));
    grid.set("vector", builder.getI64ArrayAttr(vector));

    setMetricsEntry(funcOp, "tile_grid",
                    grid.getDictionary(funcOp.getContext()));
  }
};

}  // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createRecordTileGridPass() {
  return std::make_unique<RecordTileGridPass>();
}

void registerRecordTileGridPass() { PassRegistration<RecordTileGridPass>(); }

}  // namespace mlir::coralnpu_compiler
