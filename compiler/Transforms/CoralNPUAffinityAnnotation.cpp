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

#include <algorithm>
#include <cmath>

#include "compiler/Target/Utils.h"
#include "compiler/Transforms/CoralNPUTileSizeSelectionUtils.h"
#include "compiler/Transforms/DeviceUtils.h"
#include "compiler/Transforms/Passes.h"

// IREE
#include "iree/compiler/Codegen/Utils/CPUUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"

// MLIR
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Pass/Pass.h"

// LLVM
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace mlir::iree_compiler;

namespace mlir::coralnpu_compiler {

#define GEN_PASS_DEF_CORALNPUAFFINITYANNOTATION
#include "compiler/Transforms/Passes.h.inc"

namespace {

bool isSupportedType(
    Type type, const IREE::HAL::TargetBackend::SupportedTypes &supportedTypes) {
  // Shaped types must have static shapes and a supported element type.
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

FailureOr<uint64_t> estimateBytesForType(Type type) {
  // NB: pay attention to sub-byte types (i.e. divide by 8 only at the very
  // end).

  Type elementType = getElementTypeOrSelf(type);
  uint64_t bits = 0;
  // Determine the bitwidth of the (element) type.
  if (auto indexType = llvm::dyn_cast<IndexType>(elementType)) {
    bits = indexType.kInternalStorageBitWidth;
  } else if (elementType.isIntOrFloat()) {
    bits = elementType.getIntOrFloatBitWidth();
  } else {
    return failure();
  }

  // Compute bit size for shaped types from element count and bitwidth.
  if (auto shapedType = dyn_cast<ShapedType>(type)) {
    // Dynamic shapes have unbounded size.
    if (!shapedType.hasStaticShape()) return failure();

    bits *= shapedType.getNumElements();
  }

  return llvm::divideCeil(bits, 8);
}

FailureOr<uint64_t> estimateIOBytes(
    IREE::Flow::DispatchWorkgroupsOp workgroupsOp) {
  uint64_t totalBytes = 0;

  // Accumulate the byte size of all dispatch input arguments.
  for (Value arg : workgroupsOp.getArguments()) {
    auto bytes = estimateBytesForType(arg.getType());
    if (failed(bytes)) return failure();
    totalBytes += *bytes;
  }

  // Accumulate the byte size of all dispatch results.
  for (Value result : workgroupsOp.getResults()) {
    auto bytes = estimateBytesForType(result.getType());
    if (failed(bytes)) return failure();
    totalBytes += *bytes;
  }

  return totalBytes;
}

// Iterations x non-cast body ops (at least one); casts are skipped so mixed
// precision is not inflated.
double estimateComputeOps(linalg::LinalgOp linalgOp) {
  double ops = 1.0;
  for (int64_t range : linalgOp.getStaticLoopRanges()) {
    if (range <= 0) return 0.0;  // Dynamic or empty.
    ops *= range;
  }
  int64_t opsPerIter =
      llvm::count_if(linalgOp.getBlock()->without_terminator(),
                     [](Operation &op) { return !isa<CastOpInterface>(&op); });
  return ops * std::max<int64_t>(1, opsPerIter);
}

// Fraction of Zvt 16x16 accumulator tiles filled by trailing M/N dims.
double zvtTileUtilization(linalg::LinalgOp op) {
  auto acc = cast<ShapedType>(op.getDpsInits()[0].getType());
  double utilization = 1.0;
  for (int64_t d : acc.getShape().take_back(2))
    utilization *= double(d) / llvm::alignTo(d, 16);
  return utilization;
}

// Returns true if any result of the reduction fails to map to exactly one
// combiner (e.g. argmax).
bool isScalarReduction(Operation *op) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp || linalgOp.getNumReductionLoops() == 0) return false;
  Operation *yield = linalgOp.getBlock()->getTerminator();
  return !llvm::all_of(
      llvm::zip(linalgOp.getRegionOutputArgs(), yield->getOperands()),
      [](auto pair) {
        auto [arg, res] = pair;
        return arg.hasOneUse() && *arg.user_begin() == res.getDefiningOp() &&
               linalg::getCombinerOpKind(res.getDefiningOp());
      });
}

// Greedy roofline placement for the unified-memory SoC: CoralNPU if
// T_cpu / T_npu >= threshold; inputs/outputs on the other device cost a hop.
// Constants: see README.md.
// TODO: calibrate against profiled dispatches.
bool preferCoralNPU(IREE::Flow::DispatchWorkgroupsOp workgroupsOp,
                    linalg::LinalgOp rootOp, ArrayRef<Operation *> computeOps,
                    Attribute npuAffinity, double speedupThreshold, bool hasZvt,
                    int64_t vlenBits) {
  constexpr double kNpuGHz = 0.8;
  constexpr double kNpuZvtGflops = 64 * 2 * kNpuGHz;  // 64 FP32 MAC PEs.
  // RVV: two FALUs or MAC units, each writing VLEN result bits per cycle.
  const double npuRvvGflops = 2 * vlenBits / 32.0 * 2 * kNpuGHz;
  constexpr double kNpuGBps = 16 * kNpuGHz;  // 128-bit m_axi peak.
  constexpr double kNpuLaunchNs = 800.0;
  constexpr double kCpuGflops = 8.0;
  constexpr double kCpuGBps = 4.0;
  constexpr double kHopNs = 5000.0;  // Semaphore wait per cross-device hop.

  FailureOr<uint64_t> bytes = estimateIOBytes(workgroupsOp);
  if (failed(bytes)) return false;
  const bool isZvt = hasZvt && isZvtMatrixContractionShape(rootOp);
  // Above 64 MiB (Zvt) or 16 MiB of I/O, the m_axi peak overstates CoralNPU;
  // keep such dispatches on the host (limits tuned on Gemma 4).
  if (speedupThreshold > 0.0 && *bytes > (isZvt ? 64 : 16) * 1024 * 1024)
    return false;
  DenseSet<Operation *> npuHops, cpuHops;
  SmallVector<Operation *> wl;
  auto visit = [&](Operation *op) {
    if (isa_and_nonnull<IREE::Flow::TensorReshapeOp,
                        IREE::Flow::TensorUpdateOp>(op))
      wl.push_back(op);
    else if (isa_and_nonnull<IREE::Flow::DispatchWorkgroupsOp>(op))
      if (Attribute a = op->getAttr("stream.affinity"))
        (a == npuAffinity ? cpuHops : npuHops).insert(op);
  };
  for (wl = {workgroupsOp}; !wl.empty();)
    for (Value v : wl.pop_back_val()->getOperands()) visit(v.getDefiningOp());
  for (wl = {workgroupsOp}; !wl.empty();)
    for (Operation *u : wl.pop_back_val()->getUsers()) visit(u);
  if (Attribute cur = workgroupsOp->getAttr("stream.affinity"))
    if (cur == npuAffinity ? npuHops.size() <= cpuHops.size()
                           : cpuHops.size() <= npuHops.size())
      return cur == npuAffinity;
  double ops = 0.0;
  for (Operation *op : computeOps) {
    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op))
      ops += estimateComputeOps(linalgOp);
  }

  // The narrowest root input sets the tier: 16-bit runs 2x and 8-bit 4x
  // faster; the host has no BF16 extension, so bf16 runs at the f32 rate there.
  unsigned bitWidth = 32;
  bool hasBF16Input = false;
  for (Value input : rootOp.getDpsInputs()) {
    Type type = getElementTypeOrSelf(isZvt ? getZvtLoadSource(input) : input);
    if (type.isIntOrFloat())
      bitWidth = std::min(bitWidth, type.getIntOrFloatBitWidth());
    hasBF16Input |= type.isBF16();
  }
  double npuScale = bitWidth <= 8 ? 4.0 : (bitWidth <= 16 ? 2.0 : 1.0);
  double cpuScale = hasBF16Input ? 1.0 : npuScale;
  // RVV instead follows the widest root element, with floats at the f32 rate.
  unsigned rvvBitWidth = 8;
  for (Type type : rootOp->getOperandTypes()) {
    type = getElementTypeOrSelf(type);
    if (type.isIntOrFloat())
      rvvBitWidth =
          std::max(rvvBitWidth,
                   isa<FloatType>(type) ? 32u : type.getIntOrFloatBitWidth());
  }
  double npuGflops = isZvt
                         ? kNpuZvtGflops * npuScale * zvtTileUtilization(rootOp)
                         : npuRvvGflops * 32 / rvvBitWidth;

  // Latencies in ns.
  double npuNs = kNpuLaunchNs + npuHops.size() * kHopNs +
                 std::max(ops / npuGflops, *bytes / kNpuGBps);
  double cpuNs = cpuHops.size() * kHopNs +
                 std::max(ops / cpuScale / kCpuGflops, *bytes / kCpuGBps);
  return cpuNs >= speedupThreshold * npuNs;
}

struct CoralNPUAffinityAnnotationPass
    : public impl::CoralNPUAffinityAnnotationBase<
          CoralNPUAffinityAnnotationPass> {
  using CoralNPUAffinityAnnotationBase::CoralNPUAffinityAnnotationBase;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    MLIRContext *context = &getContext();

    if (!std::isfinite(rooflineSpeedupThreshold) ||
        rooflineSpeedupThreshold < 0.0) {
      moduleOp.emitError(
          "roofline-speedup-threshold must be a finite, non-negative value");
      return signalPassFailure();
    }

    FailureOr<DeviceAffinities> affinities = lookupDeviceAffinities(moduleOp);
    if (failed(affinities)) {
      return signalPassFailure();
    }

    // Nothing to annotate if no CoralNPU device is targeted.
    if (!affinities->coralnpu) return;

    DictionaryAttr targetConfig = affinities->coralnpuTargetConfig;
    const bool hasZvt = targetConfig && hasFeature(targetConfig, "+zvtbase");
    const int64_t vlenBits =
        getVlenFromTargetFeatures(targetConfig).value_or(128);

    IREE::HAL::TargetBackend::SupportedTypes supportedTypes =
        getCoralNPUSupportedTypes(context);
    SmallVector<std::function<void()>> refine;

    moduleOp.walk([&](IREE::Flow::DispatchWorkgroupsOp workgroupsOp) {
      // If op already has affinity, don't change it
      if (workgroupsOp->hasAttr("stream.affinity")) return;

      SmallVector<Operation *> computeOps;
      workgroupsOp.getWorkgroupBody().walk([&](Operation *op) {
        // Collect all compute operations in the dispatch body.
        if (iree_compiler::isComputeOp(op)) {
          computeOps.push_back(op);
        }
      });

      FailureOr<Operation *> rootOp = getRootOperation(computeOps);
      bool canRunOnNPU =
          succeeded(rootOp) && *rootOp && isa<linalg::LinalgOp>(*rootOp) &&
          isSupportedOperandAndResultTypes(*rootOp, supportedTypes) &&
          !isScalarReduction(*rootOp);

      // Route eligible dispatches by the roofline model, or fall back to host.
      if (canRunOnNPU) {
        // Warn if any non-root compute operation has unsupported types.
        if (llvm::any_of(computeOps, [&](Operation *op) {
              return op != *rootOp &&
                     !isSupportedOperandAndResultTypes(op, supportedTypes);
            })) {
          workgroupsOp.emitWarning(
              "non-root linalg operation has types not supported by CoralNPU "
              "in a dispatch eligible for CoralNPU");
        }

        refine.emplace_back([=, ops = std::move(computeOps)] {
          bool npu =
              !affinities->host ||
              preferCoralNPU(workgroupsOp, cast<linalg::LinalgOp>(*rootOp), ops,
                             affinities->coralnpu, rooflineSpeedupThreshold,
                             hasZvt, vlenBits);
          workgroupsOp->setAttr("stream.affinity",
                                npu ? affinities->coralnpu : affinities->host);
        })();
        return;
      }
      if (affinities->host) {
        workgroupsOp->setAttr("stream.affinity", affinities->host);
      }
    });

    // Re-evaluate in reverse, then forward, with consumer hops populated.
    for (auto &fn : llvm::reverse(refine)) fn();
    for (auto &fn : refine) fn();
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
