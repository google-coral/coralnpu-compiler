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

#ifndef CORALNPU_COMPILER_TOOLS_CHECK_GEN_PARSE_UTILS_H_
#define CORALNPU_COMPILER_TOOLS_CHECK_GEN_PARSE_UTILS_H_

// MLIR headers
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"

// LLVM headers
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"

// Standard C/C++ headers
#include <string>
#include <vector>

namespace mlir::check_gen {

// Parses a single instance string of parenthesized groups, e.g., "(4,8)(8,4)"
std::vector<std::vector<int64_t>> parseInstance(llvm::StringRef raw);

// Parses all raw instance strings from CLI.
std::vector<std::vector<std::vector<int64_t>>> parseAllInstances(
    const llvm::cl::list<std::string> &rawInstances);

// Parses result type from IREE ABI declaration string.
Type parseTypeFromDecl(llvm::StringRef decl, MLIRContext *context);

}  // namespace mlir::check_gen

#endif  // CORALNPU_COMPILER_TOOLS_CHECK_GEN_PARSE_UTILS_H_
