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

#include "compiler/Target/RegisterUsageTrackerPass.h"

#include <algorithm>
#include <cctype>
#include <map>
#include <set>
#include <vector>

#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace mlir::coralnpu_compiler {
namespace {

bool isVectorRegister(StringRef regName) {
  return regName.starts_with_insensitive("v") && regName.size() > 1 &&
         std::isdigit(regName[1]);
}

bool isVectorSpillSlot(const MachineInstr &instr, int fi,
                       const MachineFrameInfo &mfi,
                       const TargetRegisterInfo *regInfo) {
  if (mfi.getObjectSize(fi) >= 16) {
    return true;
  }
  for (const MachineOperand &mo : instr.operands()) {
    if (mo.isReg() && mo.getReg().isPhysical()) {
      StringRef rName = regInfo->getName(mo.getReg());
      if (isVectorRegister(rName)) {
        return true;
      }
    }
  }
  return false;
}

void gatherVectorRegisters(const MachineBasicBlock &block,
                           const TargetRegisterInfo *regInfo,
                           std::set<std::string> &vectorRegs,
                           std::string &location) {
  for (const MachineInstr &instr : block) {
    if (location.empty() && instr.getDebugLoc()) {
      raw_string_ostream ostream(location);
      instr.getDebugLoc().print(ostream);
    }
    for (const MachineOperand &operand : instr.operands()) {
      if (operand.isReg() && operand.getReg().isPhysical()) {
        Register reg = operand.getReg();
        StringRef regName = regInfo->getName(reg);
        if (isVectorRegister(regName)) {
          vectorRegs.insert(regName.str());
        }
      }
    }
  }
}

class RegisterUsageTracker : public MachineFunctionPass {
 public:
  // LLVM uses the address of this variable to identify the pass.
  static char ID;

  RegisterUsageTracker(RegisterAllocationReportCollector *collector)
      : MachineFunctionPass(ID), collector_(collector) {}

  StringRef getPassName() const override {
    return "CoralNPU Register Usage Tracker Pass";
  }

  void getAnalysisUsage(AnalysisUsage &analysisUsage) const override {
    analysisUsage.setPreservesAll();
    analysisUsage.addRequired<MachineLoopInfoWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(analysisUsage);
  }

  bool runOnMachineFunction(MachineFunction &function) override {
    if (!collector_) return false;

    std::string funcName = function.getName().str();
    const TargetRegisterInfo *regInfo =
        function.getSubtarget().getRegisterInfo();
    auto *loopInfo = &getAnalysis<MachineLoopInfoWrapperPass>().getLI();

    // 1. Function-level stats (for fallback/summary)
    std::set<std::string> funcVectorRegs;
    std::string funcLocation = "";
    for (const MachineBasicBlock &block : function) {
      gatherVectorRegisters(block, regInfo, funcVectorRegs, funcLocation);
    }
    if (!funcVectorRegs.empty()) {
      collector_->addGlobalRegs(funcName, funcVectorRegs, funcLocation);
    }

    // 2. Loop-level stats using MachineLoopInfo
    std::map<MachineLoop *, std::set<std::string>> loopRegs;
    std::map<MachineLoop *, std::string> loopLocs;

    for (const MachineBasicBlock &block : function) {
      if (MachineLoop *loop = loopInfo->getLoopFor(&block)) {
        std::string location = "";
        std::set<std::string> regs;
        gatherVectorRegisters(block, regInfo, regs, location);

        // Propagate registers to this loop and all its parent loops
        for (MachineLoop *parentLoop = loop; parentLoop;
             parentLoop = parentLoop->getParentLoop()) {
          loopRegs[parentLoop].insert(regs.begin(), regs.end());
          // Propagate location as well if parent doesn't have one yet
          if (!location.empty() && loopLocs[parentLoop].empty()) {
            loopLocs[parentLoop] = location;
          }
        }
      }
    }

    const TargetInstrInfo *tii = function.getSubtarget().getInstrInfo();
    const MachineFrameInfo &mfi = function.getFrameInfo();

    // The nullptr entry in the following maps represents the function level
    // values.
    std::map<MachineLoop *, int> loopVecSpills, loopVecReloads;
    std::map<MachineLoop *, bool> loopHasScalarSpills;

    for (const MachineBasicBlock &block : function) {
      MachineLoop *loop = loopInfo->getLoopFor(&block);
      // When loop is nullptr we record information in the function level.
      for (const MachineInstr &instr : block) {
        int fi = 0;
        if (tii->isStoreToStackSlot(instr, fi) &&
            mfi.isSpillSlotObjectIndex(fi)) {
          if (isVectorSpillSlot(instr, fi, mfi, regInfo)) {
            loopVecSpills[loop]++;
          } else {
            loopHasScalarSpills[loop] = true;
          }
        } else if (tii->isLoadFromStackSlot(instr, fi) &&
                   mfi.isSpillSlotObjectIndex(fi)) {
          if (isVectorSpillSlot(instr, fi, mfi, regInfo)) {
            loopVecReloads[loop]++;
          } else {
            loopHasScalarSpills[loop] = true;
          }
        }
      }
    }

    collector_->recordFunctionStats(funcName, loopVecSpills[nullptr],
                                    loopVecReloads[nullptr],
                                    loopHasScalarSpills[nullptr]);

    for (const auto &[loop, regs] : loopRegs) {
      const MachineBasicBlock *header = loop->getHeader();
      std::string location = loopLocs[loop];
      collector_->addLoopRegs(funcName, header, regs, location,
                              loop->getLoopDepth());
      collector_->recordLoopStats(funcName, header, loopVecSpills[loop],
                                  loopVecReloads[loop],
                                  loopHasScalarSpills[loop]);
    }

    return false;
  }

  RegisterAllocationReportCollector *collector_;
};

char RegisterUsageTracker::ID = 0;

}  // namespace

MachineFunctionPass *createRegisterUsageTrackerPass(
    RegisterAllocationReportCollector *collector) {
  return new RegisterUsageTracker(collector);
}

}  // namespace mlir::coralnpu_compiler
