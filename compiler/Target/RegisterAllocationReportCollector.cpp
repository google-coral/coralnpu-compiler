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

#include "compiler/Target/RegisterAllocationReportCollector.h"

#include <algorithm>
#include <cassert>
#include <vector>

#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::coralnpu_compiler {

namespace {

int getPhysRegCount(const std::string &regName) {
  size_t pos = regName.find_last_of("Mm");
  if (pos != std::string::npos && pos + 1 < regName.size()) {
    std::string suffix = regName.substr(pos + 1);
    if (suffix == "2") return 2;
    if (suffix == "4") return 4;
    if (suffix == "8") return 8;
    if (!suffix.empty() && (suffix[0] == 'f' || suffix[0] == 'F')) {
      return 1;
    }
  }
  return 1;
}

std::string getBlockName(const llvm::MachineBasicBlock *bb) {
  assert(bb != nullptr && "headerBB should not be null");
  std::string name;
  llvm::raw_string_ostream rso(name);
  bb->printAsOperand(rso, false);
  return name;
}

}  // namespace

void RegisterAllocationReportCollector::addSpills(const std::string &funcName,
                                                  int spills, int reloads,
                                                  int copies) {
  auto &fStats = stats[funcName];
  fStats.name = funcName;
  fStats.spills = spills;
  fStats.reloads = reloads;
  fStats.copies = copies;
}

void RegisterAllocationReportCollector::addGlobalRegs(
    const std::string &funcName, const std::set<std::string> &regs,
    const std::string &location) {
  auto &fStats = stats[funcName];
  fStats.name = funcName;
  fStats.globalVectorRegs = regs;
  fStats.globalLocation = location;
}

void RegisterAllocationReportCollector::addLoopSpills(
    const std::string &funcName, const llvm::MachineBasicBlock *headerBB,
    int spills, int reloads, int copies) {
  auto &fStats = stats[funcName];
  fStats.name = funcName;
  std::string headerName = getBlockName(headerBB);
  auto &lStats = fStats.loops[headerName];
  lStats.headerName = headerName;
  lStats.headerNum = headerBB->getNumber();
  lStats.spills = spills;
  lStats.reloads = reloads;
  lStats.copies = copies;
}

void RegisterAllocationReportCollector::addLoopRegs(
    const std::string &funcName, const llvm::MachineBasicBlock *headerBB,
    const std::set<std::string> &regs, const std::string &location, int depth) {
  assert(headerBB != nullptr && "headerBB should not be null");
  auto &fStats = stats[funcName];
  fStats.name = funcName;
  std::string headerName = getBlockName(headerBB);
  auto &lStats = fStats.loops[headerName];
  lStats.headerName = headerName;
  lStats.headerNum = headerBB->getNumber();
  lStats.depth = depth;
  lStats.vectorRegs = regs;
  lStats.location = location;
}

void RegisterAllocationReportCollector::dumpJson(
    llvm::raw_ostream &os, const std::string &filterPattern) const {
  llvm::Regex filterRegex;
  bool hasFilter = !filterPattern.empty();
  if (hasFilter) {
    std::string regexError;
    filterRegex = llvm::Regex("^(" + filterPattern + ")$");
    if (!filterRegex.isValid(regexError)) {
      llvm::errs() << "Warning: invalid regex pattern '" << filterPattern
                   << "', disabling filter. Error: " << regexError << "\n";
      hasFilter = false;
    }
  }

  llvm::json::OStream J(os, 2);
  J.object([&] {
    J.attributeArray("dispatches", [&] {
      for (const auto &pair : stats) {
        const auto &fStats = pair.second;
        if (hasFilter && !filterRegex.match(fStats.name)) {
          continue;
        }
        if (fStats.spills == 0 && fStats.reloads == 0 && fStats.copies == 0 &&
            fStats.loops.empty() && fStats.globalVectorRegs.empty()) {
          continue;
        }
        J.object([&] {
          J.attribute("name", fStats.name);
          J.attribute("spills", fStats.spills);
          J.attribute("reloads", fStats.reloads);
          J.attribute("copies", fStats.copies);
          if (!fStats.globalVectorRegs.empty()) {
            J.attribute("global_location", fStats.globalLocation);
            int globalPhysRegs = 0;
            for (const auto &reg : fStats.globalVectorRegs) {
              globalPhysRegs += getPhysRegCount(reg);
            }
            J.attribute("global_vector_registers_count", globalPhysRegs);
            J.attributeArray("global_vector_registers", [&] {
              for (const auto &reg : fStats.globalVectorRegs) {
                J.value(reg);
              }
            });
          }
          J.attributeArray("loops", [&] {
            std::vector<LoopAllocStats> sortedLoops;
            for (const auto &[name, lStats] : fStats.loops) {
              sortedLoops.push_back(lStats);
            }
            std::sort(sortedLoops.begin(), sortedLoops.end(),
                      [](const LoopAllocStats &a, const LoopAllocStats &b) {
                        return a.headerNum < b.headerNum;
                      });

            for (const auto &lStats : sortedLoops) {
              J.object([&] {
                J.attribute("header", lStats.headerName);
                J.attribute("location", lStats.location);
                J.attribute("depth", lStats.depth);
                int loopPhysRegs = 0;
                for (const auto &reg : lStats.vectorRegs) {
                  loopPhysRegs += getPhysRegCount(reg);
                }
                J.attribute("vector_registers_used_count", loopPhysRegs);
                J.attributeArray("vector_registers_used", [&] {
                  for (const auto &reg : lStats.vectorRegs) {
                    J.value(reg);
                  }
                });
                J.attribute("spills", lStats.spills);
                J.attribute("reloads", lStats.reloads);
                J.attribute("copies", lStats.copies);
              });
            }
          });
        });
      }
    });
  });
  os << "\n";
}

void RegisterAllocationReportCollector::dumpPretty(
    llvm::raw_ostream &os, const std::string &filterPattern) const {
  llvm::Regex filterRegex;
  bool hasFilter = !filterPattern.empty();
  if (hasFilter) {
    std::string regexError;
    filterRegex = llvm::Regex("^(" + filterPattern + ")$");
    if (!filterRegex.isValid(regexError)) {
      llvm::errs() << "Warning: invalid regex pattern '" << filterPattern
                   << "', disabling filter. Error: " << regexError << "\n";
      hasFilter = false;
    }
  }

  os << "======================================================================"
        "==\n";
  os << "Register Allocation Report:\n";
  os << "======================================================================"
        "==\n";
  for (const auto &pair : stats) {
    const auto &fStats = pair.second;
    if (hasFilter && !filterRegex.match(fStats.name)) {
      continue;
    }
    if (fStats.spills == 0 && fStats.reloads == 0 && fStats.copies == 0 &&
        fStats.loops.empty() && fStats.globalVectorRegs.empty()) {
      continue;
    }
    os << "Dispatch: " << fStats.name << "\n";
    os << "  Total Spills: " << fStats.spills
       << ", Total Reloads: " << fStats.reloads
       << ", Total Copies: " << fStats.copies << "\n";

    if (!fStats.globalVectorRegs.empty()) {
      os << "  Function-level (non-loop):\n";
      if (!fStats.globalLocation.empty()) {
        os << "    Location: " << fStats.globalLocation << "\n";
      }
      int globalPhysRegs = 0;
      for (const auto &reg : fStats.globalVectorRegs) {
        globalPhysRegs += getPhysRegCount(reg);
      }
      os << "    Vector Registers Used: " << globalPhysRegs << " [";
      bool first = true;
      for (const auto &reg : fStats.globalVectorRegs) {
        if (!first) os << ", ";
        first = false;
        os << reg;
      }
      os << "]\n";
    }

    std::vector<LoopAllocStats> sortedLoops;
    for (const auto &[name, lStats] : fStats.loops) {
      sortedLoops.push_back(lStats);
    }
    std::sort(sortedLoops.begin(), sortedLoops.end(),
              [](const LoopAllocStats &a, const LoopAllocStats &b) {
                return a.headerNum < b.headerNum;
              });

    for (const auto &lStats : sortedLoops) {
      os << "  Loop (depth " << lStats.depth << ") at " << lStats.headerName;
      if (!lStats.location.empty()) {
        os << " (" << lStats.location << ")";
      }
      os << ":\n";

      int loopPhysRegs = 0;
      for (const auto &reg : lStats.vectorRegs) {
        loopPhysRegs += getPhysRegCount(reg);
      }
      os << "    Vector Registers Used: " << loopPhysRegs << " [";
      bool first = true;
      for (const auto &reg : lStats.vectorRegs) {
        if (!first) os << ", ";
        first = false;
        os << reg;
      }
      os << "]\n";

      os << "    Spills: " << lStats.spills << ", Reloads: " << lStats.reloads
         << ", Copies: " << lStats.copies << "\n";
    }
    os << "--------------------------------------------------------------------"
          "----\n";
  }
}

}  // namespace mlir::coralnpu_compiler
