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

#include "metrics/compiler/RecordZvtLoweringPass.h"

#include "compiler/Target/Utils.h"
#include "metrics/compiler/MetricsAttr.h"

// MLIR:
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/PassRegistry.h"

// Standard library:
#include <memory>

namespace mlir::coralnpu_compiler {

namespace {

// Returns true if |funcOp| has a Zvt matrix multiply.
bool hasZvtMatmul(Operation *funcOp) {
  return funcOp
      ->walk([](LLVM::InlineAsmOp asmOp) {
        return containsZvtMatmulMnemonic(asmOp.getAsmString())
                   ? WalkResult::interrupt()
                   : WalkResult::advance();
      })
      .wasInterrupted();
}

class RecordZvtLoweringPass final
    : public PassWrapper<RecordZvtLoweringPass,
                         InterfacePass<FunctionOpInterface>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RecordZvtLoweringPass)

  StringRef getArgument() const override {
    return "coralnpu-metrics-record-zvt-lowering";
  }

  StringRef getDescription() const override {
    return "Records whether the matrix codegen lowered each dispatch to Zvt "
           "in its coralnpu.metrics attribute";
  }

  // Records whether the dispatch function has a Zvt matrix multiply.
  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();
    if (!hasZvtTargetFeature(funcOp)) return;
    const bool lowered = hasZvtMatmul(funcOp);
    auto zvtAttr =
        dyn_cast_if_present<DictionaryAttr>(getMetricsEntry(funcOp, "zvt"));
    // Without a recorded candidate, only a lowered dispatch is of interest.
    if (!zvtAttr && !lowered) return;

    Builder builder(funcOp.getContext());
    NamedAttrList zvt;
    if (zvtAttr) {
      zvt.append(zvtAttr.getValue());
    } else {
      zvt.set("candidate", builder.getBoolAttr(false));
    }
    zvt.set("lowered", builder.getBoolAttr(lowered));
    setMetricsEntry(funcOp, "zvt", zvt.getDictionary(funcOp.getContext()));
    if (!lowered) {
      if (auto gridAttr = dyn_cast_if_present<DictionaryAttr>(
              getMetricsEntry(funcOp, "tile_grid"))) {
        NamedAttrList grid(gridAttr.getValue());
        if (grid.erase("zvt_grid")) {
          setMetricsEntry(funcOp, "tile_grid",
                          grid.getDictionary(funcOp.getContext()));
        }
      }
    }
  }
};

}  // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>>
createRecordZvtLoweringPass() {
  return std::make_unique<RecordZvtLoweringPass>();
}

void registerRecordZvtLoweringPass() {
  PassRegistration<RecordZvtLoweringPass>();
}

}  // namespace mlir::coralnpu_compiler
