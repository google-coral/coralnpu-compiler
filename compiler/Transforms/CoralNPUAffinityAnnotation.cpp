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

// IREE
#include "iree/compiler/Dialect/HAL/Analysis/DeviceAnalysis.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"

// MLIR
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"

// LLVM
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace mlir::iree_compiler;

namespace mlir::coralnpu_compiler {

#define GEN_PASS_DEF_CORALNPUAFFINITYANNOTATION
#include "compiler/Transforms/Passes.h.inc"

namespace {

bool isSupportedComputeOp(Operation *op) {
  if (isa<linalg::MatmulOp, linalg::BatchMatmulOp, linalg::Mmt4DOp>(op))
    return true;

  if (isa<linalg::Conv2DNhwcHwcfOp>(op)) return true;

  return false;
}

bool isSupportedType(
    Type type, const IREE::HAL::TargetBackend::SupportedTypes &supportedTypes) {
  if (auto shapedType = dyn_cast<ShapedType>(type)) {
    return shapedType.hasStaticShape() &&
           supportedTypes.supportsElementType(shapedType.getElementType());
  }

  return supportedTypes.supportsScalarType(type);
}

bool isSupportedOperandAndResultTypes(
    Operation *op,
    const IREE::HAL::TargetBackend::SupportedTypes &supportedTypes) {
  auto isSupported = [&](Type t) { return isSupportedType(t, supportedTypes); };
  return llvm::all_of(op->getOperandTypes(), isSupported) &&
         llvm::all_of(op->getResultTypes(), isSupported);
}

int64_t estimateBytesForType(Type type) {
  // NB: pay attention to sub-byte types.

  if (auto shapedType = dyn_cast<ShapedType>(type)) {
    Type elementType = shapedType.getElementType();

    unsigned elementBits = elementType.getIntOrFloatBitWidth();
    return llvm::divideCeil(shapedType.getNumElements() * elementBits, 8);
  }

  unsigned bits = type.getIntOrFloatBitWidth();
  return llvm::divideCeil(bits, 8);
}

// op must be an operation that can execute on coralnpu, type wise
int64_t estimateIOBytes(Operation *op) {
  int64_t totalBytes = 0;

  for (Value operand : op->getOperands()) {
    totalBytes += estimateBytesForType(operand.getType());
  }

  for (Value operand : op->getResults()) {
    totalBytes += estimateBytesForType(operand.getType());
  }

  return totalBytes;
}

// Discover the NPU device alias.
IREE::HAL::DeviceAffinityAttr getCoralNPUDeviceAffinityAttr(
    MLIRContext *context, ModuleOp moduleOp) {
  IREE::HAL::DeviceAnalysis deviceAnalysis(moduleOp);
  if (failed(deviceAnalysis.run())) {
    return nullptr;
  }

  for (auto globalOp : deviceAnalysis.getDeviceGlobals()) {
    auto deviceSet = deviceAnalysis.lookupDeviceTargets(globalOp);
    if (!deviceSet) continue;
    for (auto targetAttr : deviceSet->getValues()) {
      if (targetAttr.getDeviceID().getValue() == "coralnpu") {
        return IREE::HAL::DeviceAffinityAttr::get(
            context, SymbolRefAttr::get(globalOp.getGlobalName()),
            /*queue_mask=*/-1ll);
      }
    }
  }

  return nullptr;
}

struct CoralNPUAffinityAnnotationPass
    : public impl::CoralNPUAffinityAnnotationBase<
          CoralNPUAffinityAnnotationPass> {
  using CoralNPUAffinityAnnotationBase::CoralNPUAffinityAnnotationBase;

  // Return true iff the NPU can handle the computation.
  bool canExecuteOnCoralNPU(
      Operation *op,
      const IREE::HAL::TargetBackend::SupportedTypes &supportedTypes) {
    return isSupportedComputeOp(op) &&
           isSupportedOperandAndResultTypes(op, supportedTypes);
  }

  // Return true iff we think it will be beneficial for the NPU to do the
  // computation.
  bool shouldExecuteOnCoralNPU(Operation *op) {
    return ioMinThresholdBytes < estimateIOBytes(op);
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    MLIRContext *context = &getContext();

    if (ioMinThresholdBytes < 0) {
      moduleOp.emitError("io-min-threshold-bytes must be non-negative, got ")
          << ioMinThresholdBytes;
      signalPassFailure();
      return;
    }

    iree_compiler::IREE::HAL::DeviceAffinityAttr coralnpuAffinityAttr =
        getCoralNPUDeviceAffinityAttr(context, moduleOp);
    if (!coralnpuAffinityAttr) return;

    IREE::HAL::TargetBackend::SupportedTypes supportedTypes =
        getCoralNPUSupportedTypes(context);

    moduleOp.walk([&](Operation *op) {
      // If op already has affinity, don't change it
      if (op->getAttr("stream.affinity")) return;

      if (canExecuteOnCoralNPU(op, supportedTypes) &&
          shouldExecuteOnCoralNPU(op)) {
        op->setAttr("stream.affinity", coralnpuAffinityAttr);
      }
    });
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>>
createCoralNPUAffinityAnnotationPass() {
  return std::make_unique<CoralNPUAffinityAnnotationPass>();
}

std::unique_ptr<OperationPass<ModuleOp>> createCoralNPUAffinityAnnotationPass(
    CoralNPUAffinityAnnotationOptions options) {
  return std::make_unique<CoralNPUAffinityAnnotationPass>(std::move(options));
}

}  // namespace mlir::coralnpu_compiler
