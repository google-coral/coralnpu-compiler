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

#ifndef CORALNPU_COMPILER_TOOLS_CHECK_GEN_COMPILE_UTILS_H_
#define CORALNPU_COMPILER_TOOLS_CHECK_GEN_COMPILE_UTILS_H_

// IREE headers
#include "iree/compiler/ConstEval/Runtime.h"
#include "iree/compiler/embedding_api.h"

// MLIR headers
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

// Standard C/C++ headers
#include <string>
#include <vector>

namespace mlir::check_gen {

// Checks IREE compiler error, prints it if present, and destroys it.
// Returns true if there was an error.
bool checkError(iree_compiler_error_t *error);

// Writes a ModuleOp to a file.
LogicalResult writeModuleToFile(ModuleOp module, const std::string &outPath);

// Initializes the IREE compiler session and MLIR context.
// Loads required dialects (func, tensor).
iree_compiler_session_t *initCompiler(MLIRContext *&context);

// Compiles the merged module using IREE compiler session.
// Returns the output object owning the compiled binary memory (nullptr on
// error). Sets binaryData and binarySize output parameters.
iree_compiler_output_t *compileModule(iree_compiler_invocation_t *inv,
                                      ModuleOp mergedModuleOp,
                                      void *&binaryData, uint64_t &binarySize);

// Evaluates a function using ConstEval JIT.
// Returns empty vector on failure.
std::vector<TypedAttr>
evaluateFunction(mlir::iree_compiler::ConstEval::CompiledBinary &binary,
                 Location loc, StringRef funcName, ArrayRef<Type> resultTypes,
                 ArrayRef<TypedAttr> args);

} // namespace mlir::check_gen

#endif // CORALNPU_COMPILER_TOOLS_CHECK_GEN_COMPILE_UTILS_H_
