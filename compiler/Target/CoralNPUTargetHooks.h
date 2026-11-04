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

// Hooks through which code outside the CoralNPU target backend (e.g. the
// compiler metrics in metrics/compiler) observes how the backend serializes
// the executables, without the backend knowing what the hooks do.

#ifndef COMPILER_TARGET_CORALNPUTARGETHOOKS_H_
#define COMPILER_TARGET_CORALNPUTARGETHOOKS_H_

// IREE:
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"

// MLIR:
#include "mlir/Support/LLVM.h"

// LLVM:
#include "llvm/ADT/ArrayRef.h"

// Standard library:
#include <cstdint>

namespace mlir::iree_compiler::IREE::HAL {
struct LLVMTarget;
}  // namespace mlir::iree_compiler::IREE::HAL

namespace mlir::coralnpu_compiler {

// The hooks of the CoralNPU target backend.
class CoralNPUTargetHooks {
 public:
  virtual ~CoralNPUTargetHooks() = default;

  // Called after |variantOp| has been serialized for |target|. |linkedElf| is
  // the embedded firmware ELF (`target.getLinkEmbedded()`) that the backend
  // linked for a dynamic library executable, or empty for static libraries and
  // host-linked shared libraries.
  virtual void onExecutableSerialized(
      iree_compiler::IREE::HAL::ExecutableVariantOp variantOp,
      const iree_compiler::IREE::HAL::LLVMTarget &target,
      ArrayRef<uint8_t> linkedElf) {}
};

}  // namespace mlir::coralnpu_compiler

#endif  // COMPILER_TARGET_CORALNPUTARGETHOOKS_H_
