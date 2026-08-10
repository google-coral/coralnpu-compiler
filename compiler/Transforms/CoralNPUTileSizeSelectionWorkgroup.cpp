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
      signalPassFailure();
      return;
    }

    auto tilingOp = analysis.rootTilingOp;

    auto compilationInfo = getCompilationInfo(tilingOp);
    if (!compilationInfo) {
      return;
    }

    auto cpuLoweringConfig = dyn_cast<IREE::CPU::LoweringConfigAttr>(
        compilationInfo.getLoweringConfig());
    if (!cpuLoweringConfig) {
      tilingOp->emitOpError("expected CPU lowering config");
      signalPassFailure();
      return;
    }

    auto cacheParallelLevel = IREE::CPU::TilingLevel::CacheParallelTiles;
    if (!cpuLoweringConfig.hasTilingLevel(
            static_cast<unsigned>(cacheParallelLevel))) {
      tilingOp->emitOpError("missing CacheParallelTiles config");
      signalPassFailure();
      return;
    }

    auto cacheParallelAttr = cpuLoweringConfig.getTilingLevelAttr(
        static_cast<unsigned>(cacheParallelLevel));

    SmallVector<NamedAttribute> configItems(
        cpuLoweringConfig.getConfig().getValue());

    auto name = StringAttr::get(context,
                                IREE::CPU::getTilingLevelName(
                                    IREE::CPU::TilingLevel::DistributionTiles));
    bool found = false;
    for (auto &item : configItems) {
      if (item.getName() == name) {
        item.setValue(cacheParallelAttr);
        found = true;
        break;
      }
    }
    if (!found) {
      configItems.push_back(NamedAttribute(name, cacheParallelAttr));
    }

    auto newLoweringConfig =
        IREE::CPU::LoweringConfigAttr::get(context, configItems);

    auto newCompilationInfo = IREE::Codegen::CompilationInfoAttr::get(
        context, newLoweringConfig, compilationInfo.getTranslationInfo());

    setCompilationInfo(tilingOp, newCompilationInfo);

    markAnalysesPreserved<CoralNPUTileSizeSelectionAnalysis>();
  }
};

}  // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createCoralNPUTileSizeSelectionWorkgroupPass() {
  return std::make_unique<CoralNPUTileSizeSelectionWorkgroupPass>();
}

}  // namespace mlir::coralnpu_compiler
