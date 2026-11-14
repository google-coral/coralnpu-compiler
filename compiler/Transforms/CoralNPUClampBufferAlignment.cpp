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

#include "compiler/Target/Utils.h"
#include "compiler/Transforms/Passes.h"

// MLIR:
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"

// LLVM:
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"

// Standard library:
#include <cstdint>
#include <memory>
#include <optional>

namespace mlir::coralnpu_compiler {

#define GEN_PASS_DEF_CORALNPUCLAMPBUFFERALIGNMENT
#include "compiler/Transforms/Passes.h.inc"

namespace {

// Returns true if an allocation or global has an explicit byte alignment
// greater than `kBufferAlignmentBytes` (16 bytes) and its type does not
// contain a `VectorType` (which `MemRefToLLVM` lowers to naturally aligned
// 32-128 B vector loads/stores).
bool shouldClampAlignment(std::optional<uint64_t> alignment, Type type) {
  if (!alignment || *alignment <= kBufferAlignmentBytes) return false;

  return !type.walk([](VectorType) { return WalkResult::interrupt(); })
              .wasInterrupted();
}

struct CoralNPUClampBufferAlignmentPass final
    : impl::CoralNPUClampBufferAlignmentBase<CoralNPUClampBufferAlignmentPass> {
  void runOnOperation() override {
    // Temporary CoralNPU-pipeline pass: relies on
    // `splitVectorTransfersTo != "vector-transfers"`, `enableFullUnroll()`,
    // `useVectorAlignment = false`, and `MemRefToLLVM` copying `memref.global`
    // alignment only onto `llvm.mlir.global`, so no transitive view/symbol
    // walk or `llvm.func` scan is required.
    getOperation().walk<WalkOrder::PreOrder>([](Operation *op) {
      // Skip lowered `llvm.func` bodies when running after `ConvertToLLVMPass`
      // so the post-LLVM pass only visits module-scope `llvm.mlir.global` ops.
      if (llvm::isa<LLVM::LLVMFuncOp>(op)) return WalkResult::skip();

      llvm::TypeSwitch<Operation *>(op)
          .Case<memref::AllocOp, memref::AllocaOp>([](auto allocOp) {
            if (!shouldClampAlignment(allocOp.getAlignment(),
                                      allocOp.getType())) {
              return;
            }
            // `vector.type_cast` reinterprets a scalar memref as a
            // vector-element memref whose loads/stores default to the vector's
            // natural ABI alignment.
            if (llvm::any_of(allocOp->getUsers(),
                             llvm::IsaPred<vector::TypeCastOp>)) {
              return;
            }
            allocOp.setAlignment(kBufferAlignmentBytes);
          })
          .Case<memref::GlobalOp, LLVM::GlobalOp>([](auto globalOp) {
            // Only clamp read-only `.rodata` constant globals.
            if (!globalOp.getConstant()) return;
            if (shouldClampAlignment(globalOp.getAlignment(),
                                     globalOp.getType())) {
              globalOp.setAlignment(kBufferAlignmentBytes);
            }
          });
      return WalkResult::advance();
    });
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>>
createCoralNPUClampBufferAlignmentPass() {
  return std::make_unique<CoralNPUClampBufferAlignmentPass>();
}

}  // namespace mlir::coralnpu_compiler
