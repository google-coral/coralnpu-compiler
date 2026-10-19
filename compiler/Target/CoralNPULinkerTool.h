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

#ifndef COMPILER_TARGET_CORALNPULINKERTOOL_H_
#define COMPILER_TARGET_CORALNPULINKERTOOL_H_

#include <cstdint>
#include <memory>
#include <string>

#include "compiler/plugins/target/LLVMCPU/LLVMTargetOptions.h"
#include "compiler/plugins/target/LLVMCPU/LinkerTool.h"
#include "llvm/TargetParser/Triple.h"

namespace mlir::coralnpu_compiler {

// Creates the linker tool for CoralNPU embedded ELFs. It links with the linker
// script `linkerScriptPath` (if empty, the script next to the compiler), and
// sets the size of the CoralNPU stack to `stackSizeBytes`.
std::unique_ptr<iree_compiler::IREE::HAL::LinkerTool> createCoralNPULinkerTool(
    const llvm::Triple& targetTriple,
    iree_compiler::IREE::HAL::LLVMTargetOptions& targetOptions,
    std::string linkerScriptPath, int64_t stackSizeBytes);

}  // namespace mlir::coralnpu_compiler

#endif  // COMPILER_TARGET_CORALNPULINKERTOOL_H_
