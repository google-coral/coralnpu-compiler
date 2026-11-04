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

#ifndef COMPILER_TARGET_UTILS_H_
#define COMPILER_TARGET_UTILS_H_

// IREE:
#include "iree/compiler/Dialect/HAL/Target/TargetBackend.h"

// MLIR:
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

// LLVM:
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

// Standard library:
#include <cstdint>
#include <mutex>

namespace mlir::coralnpu_compiler {

// Returns the set of scalar and element types supported by CoralNPU.
iree_compiler::IREE::HAL::TargetBackend::SupportedTypes
getCoralNPUSupportedTypes(MLIRContext *context);

// Returns the lock that keeps apart the reports that executables compiling in
// parallel write to stdout or stderr (e.g. the register allocation report).
std::mutex &getReportMutex();

// Writes a report to |dir| / |fileName|, to stdout when |dir| is "-", or to
// stderr when |dir| is empty. Stream writes hold `getReportMutex()`; file I/O
// errors emit a warning on |op| without failing compilation.
void writeReport(Operation *op, StringRef dir, StringRef fileName,
                 llvm::function_ref<void(llvm::raw_ostream &)> writeFn);

// Parses VLEN in bits from the target CPU features of |funcOp| or |config|:
// the largest N among `+zvl<N>b`.
FailureOr<int64_t> getVlenFromTargetFeatures(FunctionOpInterface funcOp);
FailureOr<int64_t> getVlenFromTargetFeatures(DictionaryAttr config);

// Returns the native vector size in bytes of the HAL executable target of |op|
// (the "native_vector_size" entry of the target config), or 16 bytes if the
// target does not give it.
int64_t getNativeVectorBytes(Operation *op);

// Returns true if |op| targets CoralNPU with |feature| (e.g. "+zvfbfmin").
bool hasCoralNPUTargetFeature(Operation *op, StringRef feature);

// Returns true if the enclosing HAL executable targets CoralNPU with +zvtbase.
bool hasZvtTargetFeature(Operation *op);

}  // namespace mlir::coralnpu_compiler

#endif  // COMPILER_TARGET_UTILS_H_
