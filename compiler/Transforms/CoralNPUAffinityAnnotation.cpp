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

#include <limits>

#include "compiler/Target/Utils.h"
#include "compiler/Transforms/Passes.h"

// IREE
#include "iree/compiler/Dialect/HAL/Analysis/DeviceAnalysis.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"

// MLIR
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
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

bool canBeVectorized(Operation *op) {
  // 2D Convolutions (no channels)
  if (isa<linalg::Conv2DOp>(op)) {
    return true;
  }

  // 2D Convolutions (NHWC)
  if (isa<linalg::Conv2DNhwcHwcfOp, linalg::Conv2DNhwcHwcfQOp,
          linalg::Conv2DNhwcFhwcOp, linalg::Conv2DNhwcFhwcQOp>(op)) {
    return true;
  }

  // 2D Convolutions (NCHW)
  if (isa<linalg::Conv2DNchwFchwOp, linalg::Conv2DNchwFchwQOp>(op)) {
    return true;
  }

  // 2D Grouped Convolutions (NHWC)
  if (isa<linalg::Conv2DNhwgcGfhwcOp, linalg::Conv2DNhwgcGfhwcQOp>(op)) {
    return true;
  }

  // 2D Grouped Convolutions (NCHW)
  if (isa<linalg::Conv2DNgchwGfchwOp, linalg::Conv2DNgchwGfchwQOp,
          linalg::Conv2DNgchwFgchwOp>(op)) {
    return true;
  }

  // 2D Depthwise Convolutions (NHWC)
  if (isa<linalg::DepthwiseConv2DNhwcHwcOp, linalg::DepthwiseConv2DNhwcHwcQOp,
          linalg::DepthwiseConv2DNhwcHwcmOp,
          linalg::DepthwiseConv2DNhwcHwcmQOp>(op)) {
    return true;
  }

  // 3D Convolutions (NDHWC & NCDHW)
  if (isa<linalg::Conv3DOp, linalg::Conv3DNdhwcDhwcfOp,
          linalg::Conv3DNdhwcDhwcfQOp, linalg::Conv3DNcdhwFcdhwOp>(op)) {
    return true;
  }

  // 3D Depthwise Convolutions (NDHWC)
  if (isa<linalg::DepthwiseConv3DNdhwcDhwcOp,
          linalg::DepthwiseConv3DNdhwcDhwcmOp>(op)) {
    return true;
  }

  // 2D Pooling (NHWC)
  if (isa<linalg::PoolingNhwcSumOp, linalg::PoolingNhwcMaxOp,
          linalg::PoolingNhwcMinOp, linalg::PoolingNhwcMaxUnsignedOp,
          linalg::PoolingNhwcMinUnsignedOp>(op)) {
    return true;
  }

  // 2D Pooling (NCHW)
  if (isa<linalg::PoolingNchwSumOp, linalg::PoolingNchwMaxOp>(op)) {
    return true;
  }

  // 3D Pooling (NDHWC)
  if (isa<linalg::PoolingNdhwcSumOp, linalg::PoolingNdhwcMaxOp,
          linalg::PoolingNdhwcMinOp>(op)) {
    return true;
  }

  // 1D Convolutions
  if (isa<linalg::Conv1DNwcWcfOp, linalg::Conv1DNcwFcwOp, linalg::Conv1DOp>(
          op)) {
    return true;
  }

  // 1D Depthwise Convolutions
  if (isa<linalg::DepthwiseConv1DNwcWcOp, linalg::DepthwiseConv1DNcwCwOp,
          linalg::DepthwiseConv1DNwcWcmOp>(op)) {
    return true;
  }

  // 1D Pooling
  if (isa<linalg::PoolingNwcSumOp, linalg::PoolingNwcMaxOp,
          linalg::PoolingNwcMaxUnsignedOp, linalg::PoolingNwcMinOp,
          linalg::PoolingNwcMinUnsignedOp, linalg::PoolingNcwSumOp,
          linalg::PoolingNcwMaxOp>(op)) {
    return true;
  }

  // Matmuls, Contractions, Matrix-Vector, and Dot
  if (isa<linalg::MatmulOp, linalg::QuantizedMatmulOp, linalg::BatchMatmulOp,
          linalg::QuantizedBatchMatmulOp, linalg::BatchReduceMatmulOp,
          linalg::ContractOp, linalg::MatvecOp, linalg::BatchMatvecOp,
          linalg::VecmatOp, linalg::BatchVecmatOp, linalg::DotOp,
          linalg::Mmt4DOp, linalg::BatchMmt4DOp>(op)) {
    return true;
  }

  // Elementwise Arithmetic Binary
  if (isa<linalg::AddOp, linalg::SubOp, linalg::MulOp, linalg::MinOp,
          linalg::MaxOp>(op)) {
    return true;
  }

  // Elementwise Division Binary
  if (isa<linalg::DivOp, linalg::DivUnsignedOp>(op)) {
    return true;
  }

  // Elementwise Unary Arithmetic
  if (isa<linalg::AbsOp, linalg::NegFOp, linalg::SquareOp, linalg::CopyOp>(
          op)) {
    return true;
  }

  // Elementwise Rounding & Truncation
  if (isa<linalg::CeilOp, linalg::FloorOp, linalg::RoundOp>(op)) {
    return true;
  }

  // Elementwise Roots & Reciprocals
  if (isa<linalg::SqrtOp, linalg::RsqrtOp, linalg::ReciprocalOp>(op)) {
    return true;
  }

  // Elementwise Transcendentals & Non-linear Math
  if (isa<linalg::ExpOp, linalg::LogOp, linalg::TanhOp, linalg::ErfOp,
          linalg::PowFOp>(op)) {
    return true;
  }

  // Elementwise Map
  if (isa<linalg::MapOp>(op)) {
    return true;
  }

  if (auto elementwiseOp = dyn_cast<linalg::ElementwiseOp>(op)) {
    auto kind = elementwiseOp.getKind();
    // TODO(sflur): make sure we actually support these, and add others
    return kind == linalg::ElementwiseKind::add ||
           kind == linalg::ElementwiseKind::div ||
           kind == linalg::ElementwiseKind::mul ||
           kind == linalg::ElementwiseKind::sub;
  }

  // Transpose
  // TODO: enable TransposeOp
  // if (isa<linalg::TransposeOp>(op)) {
  //   return true;
  // }

  // Broadcast
  if (isa<linalg::BroadcastOp>(op)) {
    return true;
  }

  // TODO(sflur): enable PackOp/UnPackOp
  // // Pack and Unpack
  // if (isa<linalg::PackOp, linalg::UnPackOp>(op)) {
  //   return true;
  // }

  // Reductions
  if (isa<linalg::ReduceOp>(op)) {
    return true;
  }

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
    if (!shapedType.hasStaticShape())
      return std::numeric_limits<int64_t>::max();
    Type elementType = shapedType.getElementType();

    unsigned elementBits = elementType.getIntOrFloatBitWidth();
    return llvm::divideCeil(shapedType.getNumElements() * elementBits, 8);
  }

  unsigned bits = 0;
  if (auto indexType = llvm::dyn_cast<IndexType>(type)) {
    bits = indexType.kInternalStorageBitWidth;
  } else {
    bits = type.getIntOrFloatBitWidth();
  }
  return llvm::divideCeil(bits, 8);
}

// op must be an operation that can execute on coralnpu, type wise
int64_t estimateIOBytes(Operation *op) {
  int64_t totalBytes = 0;

  auto inputs = isa<linalg::LinalgOp>(op)
                    ? cast<linalg::LinalgOp>(op).getDpsInputs()
                    : llvm::to_vector(op->getOperands());
  for (Value input : inputs) {
    if (!matchPattern(input, m_Constant())) {
      totalBytes += estimateBytesForType(input.getType());
    }
  }

  for (Value result : op->getResults()) {
    totalBytes += estimateBytesForType(result.getType());
  }

  return totalBytes;
}

struct CoralNPUAffinityAnnotationPass
    : public impl::CoralNPUAffinityAnnotationBase<
          CoralNPUAffinityAnnotationPass> {
  using CoralNPUAffinityAnnotationBase::CoralNPUAffinityAnnotationBase;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    MLIRContext *context = &getContext();

    if (ioMinThresholdBytes < 0) {
      moduleOp.emitError("io-min-threshold-bytes must be non-negative, got ")
          << ioMinThresholdBytes;
      signalPassFailure();
      return;
    }

    IREE::HAL::DeviceAnalysis deviceAnalysis(moduleOp);
    if (failed(deviceAnalysis.run())) {
      moduleOp.emitWarning("failed to run device analysis");
      return;
    }

    IREE::HAL::DeviceAffinityAttr coralnpuAffinityAttr = nullptr;
    IREE::HAL::DeviceAffinityAttr hostAffinityAttr = nullptr;

    for (auto globalOp : deviceAnalysis.getDeviceGlobals()) {
      if (coralnpuAffinityAttr && hostAffinityAttr) break;
      auto deviceSet = deviceAnalysis.lookupDeviceTargets(globalOp);
      if (!deviceSet) continue;
      for (auto targetAttr : deviceSet->getValues()) {
        auto deviceAffinity = IREE::HAL::DeviceAffinityAttr::get(
            context, SymbolRefAttr::get(globalOp.getGlobalName()),
            /*queue_mask=*/-1ll);
        if (targetAttr.getDeviceID().getValue() == "coralnpu") {
          if (!coralnpuAffinityAttr) coralnpuAffinityAttr = deviceAffinity;
        } else if (!hostAffinityAttr) {
          hostAffinityAttr = deviceAffinity;
        }
      }
    }

    if (!coralnpuAffinityAttr) return;

    if (auto defaultAffinity =
            dyn_cast_if_present<IREE::HAL::DeviceAffinityAttr>(
                IREE::Stream::AffinityAttr::lookupOrDefault(moduleOp))) {
      if (defaultAffinity != coralnpuAffinityAttr) {
        hostAffinityAttr = defaultAffinity;
      }
    }

    if (hostAffinityAttr) {
      if (!moduleOp->hasAttr("stream.topology")) {
        auto host = hostAffinityAttr.getDevice(),
             npu = coralnpuAffinityAttr.getDevice();
        IREE::HAL::DeviceLinkAttr links[] = {
            IREE::HAL::DeviceLinkAttr::get(context, host, npu,
                                           /*unified_memory=*/true,
                                           /*transparent_access=*/true,
                                           nullptr),
            IREE::HAL::DeviceLinkAttr::get(context, npu, host,
                                           /*unified_memory=*/true,
                                           /*transparent_access=*/true,
                                           nullptr)};
        moduleOp->setAttr("stream.topology",
                          IREE::HAL::DeviceTopologyAttr::get(context, links));
      }

      if (!moduleOp->hasAttr("stream.affinity.default")) {
        moduleOp->setAttr("stream.affinity.default", hostAffinityAttr);
      }
    }

    IREE::HAL::TargetBackend::SupportedTypes supportedTypes =
        getCoralNPUSupportedTypes(context);

    moduleOp.walk([&](Operation *op) {
      // If op already has affinity, don't change it
      if (op->hasAttr("stream.affinity")) return;

      // Only assign device affinity to compute (linalg) operations. Pure
      // layout and memory operations remain unannotated to enable fusion into
      // adjacent dispatches.
      auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
      if (!linalgOp) return;

      if (isSupportedOperandAndResultTypes(op, supportedTypes) &&
          canBeVectorized(op)) {
        if (estimateIOBytes(op) > ioMinThresholdBytes) {
          op->setAttr("stream.affinity", coralnpuAffinityAttr);
        }
        // Operations that meet supported type and vectorization criteria but
        // fall below the minimum IO threshold are intentionally left
        // unannotated to allow fusion with adjacent operations, falling back to
        // stream.affinity.default if executed standalone.
      } else if (hostAffinityAttr) {
        // Operations that cannot be supported or vectorized on CoralNPU are
        // assigned to the host device fallback.
        op->setAttr("stream.affinity", hostAffinityAttr);
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
