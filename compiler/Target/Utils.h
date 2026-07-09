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

#include "iree/compiler/Dialect/HAL/Target/TargetBackend.h"
#include "mlir/IR/MLIRContext.h"

namespace mlir::coralnpu_compiler {

// Returns the set of scalar and element types supported by CoralNPU.
iree_compiler::IREE::HAL::TargetBackend::SupportedTypes
getCoralNPUSupportedTypes(MLIRContext *context);

}  // namespace mlir::coralnpu_compiler

#endif  // COMPILER_TARGET_UTILS_H_
