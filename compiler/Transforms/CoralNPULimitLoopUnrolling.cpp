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

#include <optional>

#include "compiler/Transforms/Passes.h"

// IREE
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"

// MLIR
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/WalkResult.h"

// LLVM
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "coralnpu-limit-loop-unrolling"

namespace mlir::coralnpu_compiler {

#define GEN_PASS_DEF_CORALNPULIMITLOOPUNROLLING
#include "compiler/Transforms/Passes.h.inc"

namespace {

// Annotates the loop with `llvm.loop_annotation` to either disable or fully
// unroll the loop. If an existing `llvm.loop_annotation` is present, it updates
// the `unroll` field while preserving other fields.
void setLoopUnrollAttr(scf::ForOp forOp, bool fullUnroll) {
  MLIRContext *context = forOp.getContext();
  auto disableAttr = BoolAttr::get(context, !fullUnroll);
  auto fullAttr = fullUnroll ? BoolAttr::get(context, true) : nullptr;
  auto unrollAttr = LLVM::LoopUnrollAttr::get(
      context, disableAttr, /*count=*/nullptr, /*runtimeDisable=*/nullptr,
      /*full=*/fullAttr, /*followupUnrolled=*/nullptr,
      /*followupRemainder=*/nullptr, /*followupAll=*/nullptr);

  auto oldAnnotation =
      forOp->getAttrOfType<LLVM::LoopAnnotationAttr>("loop_annotation");

  LLVM::LoopAnnotationAttr loopAnnotation;
  if (oldAnnotation) {
    loopAnnotation = LLVM::LoopAnnotationAttr::get(
        context, oldAnnotation.getDisableNonforced(),
        oldAnnotation.getVectorize(), oldAnnotation.getInterleave(), unrollAttr,
        oldAnnotation.getUnrollAndJam(), oldAnnotation.getLicm(),
        oldAnnotation.getDistribute(), oldAnnotation.getPipeline(),
        oldAnnotation.getPeeled(), oldAnnotation.getUnswitch(),
        oldAnnotation.getMustProgress(), oldAnnotation.getIsVectorized(),
        oldAnnotation.getStartLoc(), oldAnnotation.getEndLoc(),
        oldAnnotation.getParallelAccesses());
  } else {
    loopAnnotation = LLVM::LoopAnnotationAttr::get(context,
                                                   /*disableNonforced=*/nullptr,
                                                   /*vectorize=*/nullptr,
                                                   /*interleave=*/nullptr,
                                                   /*unroll=*/unrollAttr,
                                                   /*unrollAndJam=*/nullptr,
                                                   /*licm=*/nullptr,
                                                   /*distribute=*/nullptr,
                                                   /*pipeline=*/nullptr,
                                                   /*peeled=*/nullptr,
                                                   /*unswitch=*/nullptr,
                                                   /*mustProgress=*/nullptr,
                                                   /*isVectorized=*/nullptr,
                                                   /*startLoc=*/nullptr,
                                                   /*endLoc=*/nullptr,
                                                   /*parallelAccesses=*/{});
  }
  forOp->setAttr("loop_annotation", loopAnnotation);
}

// Returns the constant value of `val` if it is a constant.
// We rely on constant folding (canonicalization) to have simplified
// arithmetic expressions of constants.
std::optional<int64_t> getConstantValue(Value val) {
  Attribute attr;
  if (matchPattern(val, m_Constant(&attr))) {
    if (auto intAttr = llvm::dyn_cast<IntegerAttr>(attr)) {
      return intAttr.getInt();
    }
  }
  return std::nullopt;
}

// Returns a constant upper bound for the loop if it can be determined.
// It handles direct constants, and min operations where at least one operand is
// constant.
std::optional<int64_t> getConstantUpperBound(scf::ForOp forOp) {
  Value ub = forOp.getUpperBound();
  if (auto constVal = getConstantValue(ub)) {
    return *constVal;
  }

  // For min operations, any constant operand serves as a valid upper bound.
  // If multiple constant operands are found, we take the minimum (tightest
  // bound).
  if (auto minOp = ub.getDefiningOp<arith::MinSIOp>()) {
    auto lhs = getConstantValue(minOp.getLhs());
    auto rhs = getConstantValue(minOp.getRhs());
    if (lhs && rhs) return std::min(*lhs, *rhs);
    if (lhs) return *lhs;
    if (rhs) return *rhs;
  }

  if (auto minOp = ub.getDefiningOp<affine::AffineMinOp>()) {
    AffineMap map = minOp.getAffineMap();
    std::optional<int64_t> minConst;
    for (AffineExpr expr : map.getResults()) {
      if (auto constExpr = llvm::dyn_cast<AffineConstantExpr>(expr)) {
        int64_t val = constExpr.getValue();
        minConst = minConst ? std::min(*minConst, val) : val;
      }
    }
    if (minConst) return minConst;
  }
  return std::nullopt;
}

// Helper to check if `val` is of the form `base + offset` where `offset` is
// constant.
bool matchAddConstant(Value val, Value base, int64_t &offset) {
  auto addOp = val.getDefiningOp<arith::AddIOp>();
  if (!addOp) return false;

  if (auto lhs = getConstantValue(addOp.getLhs())) {
    if (addOp.getRhs() == base) {
      offset = *lhs;
      return true;
    }
  }

  if (auto rhs = getConstantValue(addOp.getRhs())) {
    if (addOp.getLhs() == base) {
      offset = *rhs;
      return true;
    }
  }
  return false;
}

// Helper to check if `val` is of the form `min(..., base + limit, ...)` where
// `limit` is constant. This is typical for tiled loops: ub = min(global_ub, lb
// + tile_size).
bool matchTiledUpperBound(Value val, Value base, int64_t &limit) {
  // Case A: arith.min(..., base + limit)
  if (auto minOp = val.getDefiningOp<arith::MinSIOp>()) {
    int64_t offset = 0;
    if (matchAddConstant(minOp.getLhs(), base, offset) ||
        matchAddConstant(minOp.getRhs(), base, offset)) {
      limit = offset;
      return true;
    }
  }

  // Case B: affine.min (typically generated by tiling passes)
  if (auto minOp = val.getDefiningOp<affine::AffineMinOp>()) {
    AffineMap map = minOp.getAffineMap();
    for (auto [idx, operand] : llvm::enumerate(minOp.getOperands())) {
      if (operand != base) continue;

      for (AffineExpr expr : map.getResults()) {
        auto binaryOp = llvm::dyn_cast<AffineBinaryOpExpr>(expr);
        if (!binaryOp || binaryOp.getKind() != AffineExprKind::Add) continue;

        auto lhs = llvm::dyn_cast<AffineDimExpr>(binaryOp.getLHS());
        auto rhs = llvm::dyn_cast<AffineConstantExpr>(binaryOp.getRHS());
        if (lhs && rhs && lhs.getPosition() == idx) {
          limit = rhs.getValue();
          return true;
        }
      }
    }
  }
  return false;
}

std::optional<int64_t> getUpperBoundTripCount(scf::ForOp forOp) {
  std::optional<int64_t> lbConst = getConstantValue(forOp.getLowerBound());
  std::optional<int64_t> stepConst = getConstantValue(forOp.getStep());
  if (!stepConst) return std::nullopt;
  int64_t step = *stepConst;

  // Case 1: Constant bounds
  if (lbConst) {
    int64_t lb = *lbConst;
    std::optional<int64_t> ub = getConstantUpperBound(forOp);
    if (ub) {
      if (*ub <= lb) return 0;
      return llvm::divideCeil(*ub - lb, step);
    }
  }

  // Case 2: Tiled loop pattern: ub = min(global_ub, lb + tile_size)
  Value lb = forOp.getLowerBound();
  Value ub = forOp.getUpperBound();
  int64_t limit = 0;
  if (matchTiledUpperBound(ub, lb, limit)) {
    return llvm::divideCeil(limit, step);
  }

  return std::nullopt;
}

// Returns true if the loop has been annotated to disable unrolling.
bool isExplicitlyDisabled(scf::ForOp forOp) {
  if (auto attr =
          forOp->getAttrOfType<LLVM::LoopAnnotationAttr>("loop_annotation")) {
    if (auto unroll = attr.getUnroll()) {
      if (auto disable = unroll.getDisable()) {
        return disable.getValue();
      }
    }
  }
  return false;
}

struct CoralNPULimitLoopUnrollingPass final
    : impl::CoralNPULimitLoopUnrollingBase<CoralNPULimitLoopUnrollingPass> {
  CoralNPULimitLoopUnrollingPass() = default;
  CoralNPULimitLoopUnrollingPass(
      const CoralNPULimitLoopUnrollingOptions &options)
      : impl::CoralNPULimitLoopUnrollingBase<CoralNPULimitLoopUnrollingPass>(
            options) {}

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    // Process all loops in post-order (inner first)
    funcOp.walk<WalkOrder::PostOrder>([&](scf::ForOp forOp) {
      if (isExplicitlyDisabled(forOp)) {
        return WalkResult::advance();
      }

      // If the loop body has too many operations, unrolling it will probably
      // spill registers.
      if (llvm::range_size(forOp.getBody()->without_terminator()) > 32) {
        setLoopUnrollAttr(forOp, /*fullUnroll=*/false);
        return WalkResult::advance();
      }

      // We don't want to unroll loops with operations that are lowered to many
      // instructions, as they will overflow ITCM (llvm is not good at
      // preventing this).
      bool hasExpensiveOps = llvm::any_of(
          forOp.getBody()->without_terminator(), [](Operation &op) -> bool {
            if (op.getName().getDialectNamespace() == "math") return true;
            if (llvm::isa<scf::ForOp>(op)) return true;
            return false;
          });
      if (hasExpensiveOps) {
        setLoopUnrollAttr(forOp, /*fullUnroll=*/false);
        return WalkResult::advance();
      }

      auto tripCount = getUpperBoundTripCount(forOp);
      int64_t unrollFactor = tripCount.value_or(0);
      if (maxLoopUnrolling > 0) {
        unrollFactor =
            std::min(unrollFactor, static_cast<int64_t>(maxLoopUnrolling));
      }

      if (unrollFactor <= 1) {
        setLoopUnrollAttr(forOp, /*fullUnroll=*/false);
        return WalkResult::advance();
      }

      if (unrollFactor == *tripCount) {
        // We don't do anything (llvm will unroll the loop, as much as it thinks
        // is optimal).
        return WalkResult::advance();
      }

      // Partial unrolling
      auto result = loopUnrollByFactor(forOp, unrollFactor);
      if (failed(result)) {
        forOp->emitWarning("failed to unroll-by-factor");
        return WalkResult::advance();
      }

      if (result->mainLoopOp) {
        setLoopUnrollAttr(*result->mainLoopOp, /*fullUnroll=*/false);
      }
      if (result->epilogueLoopOp) {
        setLoopUnrollAttr(*result->epilogueLoopOp, /*fullUnroll=*/false);
      }

      return WalkResult::advance();
    });
  }
};

}  // namespace

std::unique_ptr<Pass> createCoralNPULimitLoopUnrollingPass() {
  return std::make_unique<CoralNPULimitLoopUnrollingPass>();
}

std::unique_ptr<Pass> createCoralNPULimitLoopUnrollingPass(
    CoralNPULimitLoopUnrollingOptions options) {
  return std::make_unique<CoralNPULimitLoopUnrollingPass>(options);
}

std::unique_ptr<Pass> createCoralNPULimitLoopUnrollingPass(
    int maxLoopUnrolling) {
  CoralNPULimitLoopUnrollingOptions options;
  options.maxLoopUnrolling = maxLoopUnrolling;
  return std::make_unique<CoralNPULimitLoopUnrollingPass>(options);
}

}  // namespace mlir::coralnpu_compiler
