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

#include "compiler/Transforms/DeviceUtils.h"
#include "compiler/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace mlir::coralnpu_compiler {

#define GEN_PASS_DEF_CORALNPUPROMOTEBF16ACCUMULATOR
#include "compiler/Transforms/Passes.h.inc"

namespace {

// Casts `src` to `dstElemType` via `linalg.copy` (extf/truncf).
Value createFPCast(OpBuilder &builder, Location loc, Value src,
                   Type dstElemType, ArrayRef<OpFoldResult> sizes) {
  Value init = tensor::EmptyOp::create(builder, loc, sizes, dstElemType);
  return linalg::CopyOp::create(builder, loc, src, init)->getResult(0);
}

// Returns `init` in FP32; a `linalg.fill`, also behind reshapes, is rebuilt in
// FP32.
Value createF32Accumulator(OpBuilder &builder, Location loc, Value init,
                           Type f32Type, ArrayRef<OpFoldResult> sizes) {
  Value src = init;
  while (isa_and_nonnull<tensor::ExpandShapeOp, tensor::CollapseShapeOp>(
      src.getDefiningOp()))
    src = src.getDefiningOp()->getOperand(0);
  if (auto fillOp = src.getDefiningOp<linalg::FillOp>()) {
    Value widened =
        arith::ExtFOp::create(builder, loc, f32Type, fillOp.getInputs()[0],
                              arith::FastMathFlagsAttr{});
    Value empty = tensor::EmptyOp::create(builder, loc, sizes, f32Type);
    return linalg::FillOp::create(builder, loc, widened, empty).getResult(0);
  }
  return createFPCast(builder, loc, init, f32Type, sizes);
}

// True for a generic/reduce reduction whose body is only addf/mulf on
// in-body values, so it can be cloned in FP32.
bool hasFP32ClonableBody(linalg::LinalgOp linalgOp) {
  Block *body = linalgOp.getBlock();
  return isa<linalg::ReduceOp, linalg::GenericOp>(linalgOp) &&
         linalgOp.getNumReductionLoops() > 0 &&
         !body->without_terminator().empty() &&
         llvm::all_of(body->without_terminator(), [&](Operation &op) {
           return isa<arith::AddFOp, arith::MulFOp>(op) &&
                  llvm::all_of(op.getOperands(), [&](Value v) {
                    return v.getParentBlock() == body;
                  });
         });
}

// Accumulates a BF16 reduction in FP32, rounding once at the end. With
// `hasOnlyHostDevice`, dynamic shapes stay BF16.
void promoteBF16Accumulator(linalg::LinalgOp linalgOp, IRRewriter &rewriter,
                            bool hasOnlyHostDevice) {
  Block *origBody = linalgOp.getBlock();
  if (!origBody || !linalgOp.hasPureTensorSemantics() ||
      linalgOp.getNumDpsInits() != 1)
    return;
  // Named ops rebuild their body; requiring addf skips max/min pooling.
  auto regionBuilder = linalgOp.getRegionBuilder();
  bool isNamed = regionBuilder &&
                 (linalg::isaContractionOpInterface(linalgOp) ||
                  linalg::isaConvolutionOpInterface(linalgOp)) &&
                 llvm::any_of(origBody->without_terminator(),
                              llvm::IsaPred<arith::AddFOp>);
  if (!isNamed && !hasFP32ClonableBody(linalgOp)) return;

  Value init = linalgOp.getDpsInitOperand(0)->get();
  auto initType = dyn_cast<RankedTensorType>(init.getType());
  // Host-only dynamic shapes stay BF16: IREE's data-tiled x86 bf16 matmul
  // tests (e2e_matmul_cpu_dt_inner_tiled_bf16_bf16) fail to compile otherwise.
  if (!initType || !initType.getElementType().isBF16() ||
      (hasOnlyHostDevice && linalgOp.hasDynamicShape()))
    return;
  // Unread inputs (e.g. a reduce_window's window) may have any type.
  if (!llvm::all_of(linalgOp.getDpsInputOperands(), [&](OpOperand *input) {
        return getElementTypeOrSelf(input->get().getType()).isBF16() ||
               linalgOp.getMatchingBlockArgument(input).use_empty();
      }))
    return;

  rewriter.setInsertionPoint(linalgOp);
  Location loc = linalgOp.getLoc();
  Type f32Type = rewriter.getF32Type();
  RankedTensorType accType = initType.clone(f32Type);

  SmallVector<OpFoldResult> sizes = tensor::getMixedSizes(rewriter, loc, init);
  Value accumulator = createF32Accumulator(rewriter, loc, init, f32Type, sizes);

  SmallVector<Value> newOperands(linalgOp.getDpsInputs());
  newOperands.push_back(accumulator);

  // Keep all attrs; getPrunedAttributeList() drops inherent indexing_maps.
  OperationState state(loc, linalgOp->getName().getStringRef(), newOperands,
                       TypeRange{accType}, linalgOp->getAttrs());
  Region &region = *state.addRegion();

  SmallVector<Type> blockArgTypes = llvm::map_to_vector(
      newOperands, [](Value v) { return getElementTypeOrSelf(v.getType()); });
  SmallVector<Location> blockArgLocs(newOperands.size(), loc);
  {
    OpBuilder::InsertionGuard guard(rewriter);
    Block *body = rewriter.createBlock(&region, region.end(), blockArgTypes,
                                       blockArgLocs);
    if (isNamed) {
      ImplicitLocOpBuilder bodyBuilder(loc, rewriter);
      regionBuilder(bodyBuilder, *body, state.attributes.getAttrs(),
                    [&]() { return linalgOp->emitOpError(); });
    } else {
      // Widen BF16 inputs, then clone the body with FP32 results.
      IRMapping mapping;
      for (auto [oldArg, newArg] :
           llvm::zip_equal(origBody->getArguments(), body->getArguments()))
        mapping.map(oldArg, newArg.getType().isBF16()
                                ? Value(arith::ExtFOp::create(
                                      rewriter, loc, f32Type, newArg,
                                      arith::FastMathFlagsAttr{}))
                                : newArg);
      for (Operation &op : *origBody)
        for (Value result : rewriter.clone(op, mapping)->getResults())
          result.setType(f32Type);
    }
  }

  Operation *promoted = rewriter.create(state);
  rewriter.replaceOp(linalgOp,
                     createFPCast(rewriter, loc, promoted->getResult(0),
                                  initType.getElementType(), sizes));
}

struct CoralNPUPromoteBF16AccumulatorPass
    : public impl::CoralNPUPromoteBF16AccumulatorBase<
          CoralNPUPromoteBF16AccumulatorPass> {
  using CoralNPUPromoteBF16AccumulatorBase::CoralNPUPromoteBF16AccumulatorBase;

  void runOnOperation() override {
    bool hasOnlyHostDevice = targetsOnlyHost(getOperation());
    IRRewriter rewriter(&getContext());
    getOperation()->walk([&](linalg::LinalgOp op) {
      promoteBF16Accumulator(op, rewriter, hasOnlyHostDevice);
    });
  }
};

}  // namespace

std::unique_ptr<Pass> createCoralNPUPromoteBF16AccumulatorPass() {
  return std::make_unique<CoralNPUPromoteBF16AccumulatorPass>();
}

}  // namespace mlir::coralnpu_compiler
