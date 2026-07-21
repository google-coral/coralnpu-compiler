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

#ifndef COMPILER_TARGET_CORALNPUDIAGNOSTICHANDLER_H_
#define COMPILER_TARGET_CORALNPUDIAGNOSTICHANDLER_H_

#include "compiler/Target/RegisterAllocationReportCollector.h"

// LLVM
#include "llvm/CodeGen/MachineOptimizationRemarkEmitter.h"
#include "llvm/IR/DiagnosticHandler.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Function.h"

namespace mlir::coralnpu_compiler {

class CoralNPUDiagnosticHandler : public llvm::DiagnosticHandler {
 public:
  CoralNPUDiagnosticHandler(RegisterAllocationReportCollector *collector)
      : collector_(collector) {}

  bool handleDiagnostics(const llvm::DiagnosticInfo &di) override {
    if (auto *remark =
            llvm::dyn_cast<llvm::MachineOptimizationRemarkMissed>(&di)) {
      if (remark->getPassName() == "regalloc") {
        processRegAllocRemark(*remark);
      }
    }
    // Return true to consume the diagnostic and prevent LLVM from printing it.
    // We will print it ourselves in the report if enabled.
    return true;
  }

  bool isMissedOptRemarkEnabled(llvm::StringRef passName) const override {
    return passName == "regalloc";
  }

  bool isAnyRemarkEnabled() const override { return true; }

 private:
  void processRegAllocRemark(
      const llvm::DiagnosticInfoOptimizationBase &remark) {
    if (!collector_) return;

    std::string funcName = remark.getFunction().getName().str();
    std::string remarkName = remark.getRemarkName().str();

    if (remarkName != "SpillReloadCopies" &&
        remarkName != "LoopSpillReloadCopies")
      return;

    // Cast to MIROptimization to get the MachineBasicBlock
    auto &mirRemark = llvm::cast<llvm::DiagnosticInfoMIROptimization>(remark);

    int spills = 0;
    int reloads = 0;
    int copies = 0;

    for (const auto &arg : remark.getArgs()) {
      if (arg.Key == "NumSpills") {
        llvm::StringRef(arg.Val).getAsInteger(10, spills);
      } else if (arg.Key == "NumReloads") {
        llvm::StringRef(arg.Val).getAsInteger(10, reloads);
      } else if (arg.Key == "NumVRCopies") {
        llvm::StringRef(arg.Val).getAsInteger(10, copies);
      }
    }

    if (remarkName == "SpillReloadCopies") {
      collector_->addSpills(funcName, spills, reloads, copies);
      return;
    }

    assert(remarkName == "LoopSpillReloadCopies");
    collector_->addLoopSpills(funcName, mirRemark.getBlock(), spills, reloads,
                              copies);
  }

  RegisterAllocationReportCollector *collector_;
};

}  // namespace mlir::coralnpu_compiler

#endif  // COMPILER_TARGET_CORALNPUDIAGNOSTICHANDLER_H_
