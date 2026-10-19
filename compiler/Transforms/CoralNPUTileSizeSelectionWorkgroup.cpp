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
// CoralNPU Workgroup Tile Size Selection Pass
//===----------------------------------------------------------------------===//

#include <memory>

#include "compiler/Transforms/CoralNPUTileSizeSelectionAnalysis.h"
#include "compiler/Transforms/CoralNPUTileSizeSelectionUtils.h"
#include "compiler/Transforms/Passes.h"

// IREE headers
#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUDialect.h"
#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUTypes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Utils/CPUUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"

// MLIR headers
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

// LLVM headers
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::iree_compiler;

namespace mlir::coralnpu_compiler {

#define GEN_PASS_DEF_CORALNPUTILESIZESELECTIONWORKGROUP
#include "compiler/Transforms/Passes.h.inc"

namespace {

struct CoralNPUTileSizeSelectionWorkgroupPass
    : public impl::CoralNPUTileSizeSelectionWorkgroupBase<
          CoralNPUTileSizeSelectionWorkgroupPass> {
  using CoralNPUTileSizeSelectionWorkgroupBase::
      CoralNPUTileSizeSelectionWorkgroupBase;

  void runOnOperation() override {
    auto funcOp = getOperation();
    MLIRContext *context = &getContext();

    auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(funcOp);
    if (!targetAttr || targetAttr.getBackend().getValue() != "coralnpu") {
      return;
    }

    auto &analysis = getAnalysis<CoralNPUTileSizeSelectionAnalysis>();
    if (failed(analysis.status)) {
      return;
    }

    auto tilingOp = analysis.rootTilingOp;

    // Skip the dispatch if the DTCM pass skipped it (the root op has no CPU
    // lowering config with distribution tile sizes; the DTCM pass emitted a
    // warning), or if the root op has no DTCM parallel tile sizes
    // (cache_parallel).
    auto cpuLoweringConfig =
        getLoweringConfig<IREE::CPU::LoweringConfigAttr>(tilingOp);
    auto cacheParallelLevel = IREE::CPU::TilingLevel::CacheParallelTiles;
    if (!cpuLoweringConfig || !cpuLoweringConfig.hasWorkgroupTilingLevel() ||
        !cpuLoweringConfig.hasTilingLevel(
            static_cast<unsigned>(cacheParallelLevel))) {
      return;
    }

    // Set the distribution tile sizes to the DTCM parallel tile sizes. The
    // guard above makes sure that the distribution level exists.
    NamedAttrList configItems(cpuLoweringConfig.getConfig());
    configItems.set(IREE::CPU::getTilingLevelName(
                        IREE::CPU::TilingLevel::DistributionTiles),
                    cpuLoweringConfig.getTilingLevelAttr(
                        static_cast<unsigned>(cacheParallelLevel)));
    setLoweringConfig(tilingOp,
                      IREE::CPU::LoweringConfigAttr::get(
                          context, configItems.getDictionary(context)));

    markAnalysesPreserved<CoralNPUTileSizeSelectionAnalysis>();
  }
};

}  // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createCoralNPUTileSizeSelectionWorkgroupPass() {
  return std::make_unique<CoralNPUTileSizeSelectionWorkgroupPass>();
}

}  // namespace mlir::coralnpu_compiler
