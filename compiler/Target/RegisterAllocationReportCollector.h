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

#ifndef COMPILER_TARGET_REGISTERALLOCATIONREPORTCOLLECTOR_H_
#define COMPILER_TARGET_REGISTERALLOCATIONREPORTCOLLECTOR_H_

#include <map>
#include <set>
#include <string>
#include <vector>

namespace llvm {
class MachineBasicBlock;
class raw_ostream;
}  // namespace llvm

namespace mlir::coralnpu_compiler {

// Statistics for a single loop.
struct LoopAllocStats {
  std::string headerName;
  int headerNum = 0;
  int depth = 0;
  std::string location;  // debug loc if available

  // Register utilization
  std::set<std::string> vectorRegs;

  // Spills
  int spills = 0;
  int reloads = 0;
  int copies = 0;
};

// Statistics for a function, aggregating loop stats and global stats.
struct FunctionAllocStats {
  std::string name;
  int spills = 0;  // Function total
  int reloads = 0;
  int copies = 0;

  // Function-level (non-loop) vector stats
  std::set<std::string> globalVectorRegs;
  std::string globalLocation;

  // Map loop header name to its stats
  std::map<std::string, LoopAllocStats> loops;
};

// Collects and formats register allocation statistics (spills, reloads,
// copies, and register usage) at function and loop levels.
class RegisterAllocationReportCollector {
 public:
  RegisterAllocationReportCollector() = default;

  void addSpills(const std::string &funcName, int spills, int reloads,
                 int copies);

  void addGlobalRegs(const std::string &funcName,
                     const std::set<std::string> &regs,
                     const std::string &location);

  void addLoopSpills(const std::string &funcName,
                     const llvm::MachineBasicBlock *headerBB, int spills,
                     int reloads, int copies);

  void addLoopRegs(const std::string &funcName,
                   const llvm::MachineBasicBlock *headerBB,
                   const std::set<std::string> &regs,
                   const std::string &location, int depth);

  void dumpJson(llvm::raw_ostream &os,
                const std::string &filterPattern = "") const;
  void dumpPretty(llvm::raw_ostream &os,
                  const std::string &filterPattern = "") const;

  bool isEmpty() const { return stats.empty(); }

 private:
  // Map function name to its stats
  std::map<std::string, FunctionAllocStats> stats;
};

}  // namespace mlir::coralnpu_compiler

#endif  // COMPILER_TARGET_REGISTERALLOCATIONREPORTCOLLECTOR_H_
