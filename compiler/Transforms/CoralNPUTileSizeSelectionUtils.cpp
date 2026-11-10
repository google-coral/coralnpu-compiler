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

#include "compiler/Transforms/CoralNPUTileSizeSelectionUtils.h"

// IREE:
#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUTypes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"

// MLIR:
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"

// LLVM:
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace mlir::iree_compiler;

namespace mlir::coralnpu_compiler {

void classifyLoops(TilingInterface tilingOp,
                   SmallVectorImpl<unsigned> &parallelLoops,
                   SmallVectorImpl<unsigned> &reductionLoops) {
  for (auto [index, iterType] :
       llvm::enumerate(tilingOp.getLoopIteratorTypes())) {
    switch (iterType) {
      case utils::IteratorType::parallel:
        parallelLoops.push_back(index);
        break;
      case utils::IteratorType::reduction:
        reductionLoops.push_back(index);
        break;
    }
  }
}

bool hasWideningInput(linalg::LinalgOp op) {
  if (!op || op.getNumDpsInputs() == 0 || op->getNumResults() == 0) {
    return false;
  }
  Type inTy = getElementTypeOrSelf(op.getDpsInputOperand(0)->get().getType());
  Type outTy = getElementTypeOrSelf(op->getResult(0).getType());
  if (!inTy.isIntOrFloat() || !outTy.isIntOrFloat()) return false;
  unsigned inBits = inTy.getIntOrFloatBitWidth();
  unsigned outBits = outTy.getIntOrFloatBitWidth();
  return inBits >= 8 && inBits < outBits;
}

FailureOr<linalg::ConvolutionDimensions> inferConvDims(Operation *op) {
  auto linalgOp = dyn_cast_if_present<linalg::LinalgOp>(op);
  if (!linalgOp || !linalg::isaConvolutionOpInterface(linalgOp)) {
    return failure();
  }
  if (FailureOr<linalg::ConvolutionDimensions> dims =
          linalg::inferConvolutionDims(linalgOp);
      succeeded(dims)) {
    return dims;
  }
  SmallVector<AffineMap> maps = linalgOp.getIndexingMapsArray();
  if (maps.size() < 3) return failure();
  return linalg::inferConvolutionDims(
      ArrayRef<AffineMap>{maps[0], maps[1], maps.back()});
}

FailureOr<int64_t> getVlenFromTargetFeatures(FunctionOpInterface funcOp) {
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(funcOp);
  if (!targetAttr) return failure();

  return getVlenFromTargetFeatures(targetAttr.getConfiguration());
}

FailureOr<int64_t> getVlenFromTargetFeatures(DictionaryAttr config) {
  if (!config) return failure();

  auto attr = config.getAs<StringAttr>("cpu_features");
  if (!attr) return failure();

  // +zvl<N>b implies every smaller +zvl<M>b, so VLEN is the largest N.
  int64_t vlen = 0;
  SmallVector<llvm::StringRef> features;
  attr.getValue().split(features, ',');
  for (llvm::StringRef feature : features) {
    feature = feature.trim();
    int64_t n = 0;
    if (feature.consume_front("+zvl") && feature.consume_back("b") &&
        !feature.getAsInteger(10, n))
      vlen = std::max(vlen, n);
  }
  if (vlen == 0) return failure();
  return vlen;
}

Attribute getTilingLevelAttr(MLIRContext *context, ArrayRef<int64_t> sizes) {
  SmallVector<bool> scalableFlags(sizes.size(), false);
  return IREE::Codegen::LoweringConfigTilingLevelAttr::get(
      context, sizes, /*tileInterchange=*/{}, scalableFlags);
}

int64_t getDTCMTileBudgetBytes(int64_t dtcmSizeKb, int64_t stackSizeKb,
                               int64_t reserveKb) {
  // Workgroup local memory is allocated in 4 KB pages
  // (IREE_HAL_EXECUTABLE_WORKGROUP_LOCAL_MEMORY_PAGE_SIZE /
  // LibraryBuilder::kWorkgroupLocalMemoryPageSize).
  // TODO: Share this constant when upstreamed instead of duplicating 4096.
  constexpr int64_t kBudgetGranuleBytes = 4096;
  // A negative stack or reserve would make the budget larger than the DTCM.
  if (dtcmSizeKb <= 0 || stackSizeKb < 0 || reserveKb < 0) return 0;
  int64_t freeBytes = (dtcmSizeKb - stackSizeKb - reserveKb) * 1024;
  // Less than one granule is left for the tile buffers.
  if (freeBytes < kBudgetGranuleBytes) return 0;
  return llvm::alignDown(freeBytes, kBudgetGranuleBytes);
}

bool hasPromoteOperands(Attribute loweringConfig) {
  auto cpuConfig =
      dyn_cast_or_null<IREE::CPU::LoweringConfigAttr>(loweringConfig);
  if (!cpuConfig) return false;
  auto operands = cpuConfig.getConfig().getAs<ArrayAttr>(kPromoteOperandsName);
  return operands && !operands.empty();
}

int64_t getNativeVectorBytes(Operation *op) {
  // One 128-bit register, if the target does not give the size.
  constexpr int64_t kDefaultNativeVectorBytes = 16;
  auto target = IREE::HAL::ExecutableTargetAttr::lookup(op);
  if (!target || !target.getConfiguration()) return kDefaultNativeVectorBytes;
  auto size =
      target.getConfiguration().getAs<IntegerAttr>("native_vector_size");
  return size ? size.getInt() : kDefaultNativeVectorBytes;
}

bool hasCoralNPUTargetFeature(Operation *op, StringRef feature) {
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(op);
  return targetAttr && targetAttr.getBackend().getValue() == "coralnpu" &&
         hasFeature(targetAttr.getConfiguration(), feature);
}

bool hasZvtTargetFeature(Operation *op) {
  return hasCoralNPUTargetFeature(op, "+zvtbase");
}

bool isZvtElementTypes(Type lhs, Type rhs, Type acc) {
  return lhs == rhs &&
         ((acc.isF32() && (lhs.isF32() || lhs.isBF16())) ||
          (acc.isInteger(32) && (lhs.isInteger(8) || lhs.isInteger(16))));
}

bool isZvtAccumulatorShape(int64_t m, int64_t n, bool rank1, bool swap) {
  // Rows that are not a multiple of 16 fill one partial tile, and only without
  // the swap. The asm does not support 48 rows or columns.
  if ((m % 16 != 0 && (m > 16 || swap)) || m == 48 || n == 48) return false;
  // Columns that are not a multiple of 16 are 8 or 12, in a 2-D accumulator
  // without the swap.
  if (n % 16 != 0 && ((n != 8 && n != 12) || rank1 || swap)) return false;
  // At most four 16x16 tiles.
  return llvm::alignTo(m, 16) * llvm::alignTo(n, 16) <= 1024;
}

Value getZvtLoadSource(Value input, bool strictLayoutAndSingleUse) {
  Operation *def = input.getDefiningOp();
  // The asm loads an input that no tileable op computes.
  if (!isa_and_nonnull<TilingInterface>(def)) return input;
  // Else, the producer must be an elementwise extension (`arith.extf` or
  // `arith.extsi`) of one tensor (single-consumer and 1:1 for Zvt root tiling).
  auto producer = dyn_cast<linalg::LinalgOp>(def);
  if (!producer || producer.getNumDpsInputs() != 1 ||
      producer->getNumResults() != 1 || !linalg::isElementwise(producer) ||
      (strictLayoutAndSingleUse &&
       (input.use_empty() || !llvm::all_equal(input.getUsers()) ||
        producer.getMatchingIndexingMap(producer.getDpsInputOperand(0)) !=
            producer.getMatchingIndexingMap(producer.getDpsInitOperand(0))))) {
    return nullptr;
  }
  Block *body = producer.getBlock();
  auto yieldOp = dyn_cast<linalg::YieldOp>(body->getTerminator());
  Operation *ext = yieldOp && yieldOp.getNumOperands() == 1
                       ? yieldOp.getOperand(0).getDefiningOp()
                       : nullptr;
  if (!isa_and_nonnull<arith::ExtFOp, arith::ExtSIOp>(ext) ||
      ext->getOperand(0) != body->getArgument(0) ||
      !llvm::hasSingleElement(body->without_terminator())) {
    return nullptr;
  }
  // The asm reads through one extension only: it cannot load a tensor that
  // another tileable op computes.
  Value source = producer.getDpsInputOperand(0)->get();
  if (source.getDefiningOp<TilingInterface>()) return nullptr;
  return source;
}

namespace {

// Returns true if `op` yields `acc + ext(in0) * ext(in1)` using only
// `arith.extf` or `arith.extsi` casts.
bool hasZvtMacBody(linalg::LinalgOp op) {
  Block *body = op.getBlock();
  if (!body || body->getNumArguments() != 3) return false;
  auto yield = dyn_cast<linalg::YieldOp>(body->getTerminator());
  Operation *add = yield && yield.getNumOperands() == 1
                       ? yield.getOperand(0).getDefiningOp()
                       : nullptr;
  if (!isa_and_nonnull<arith::AddFOp, arith::AddIOp>(add)) return false;
  Value acc = body->getArgument(2);
  Value prod = add->getOperand(0) == acc
                   ? add->getOperand(1)
                   : (add->getOperand(1) == acc ? add->getOperand(0) : Value());
  Operation *mul = prod ? prod.getDefiningOp() : nullptr;
  if (!isa_and_nonnull<arith::MulFOp, arith::MulIOp>(mul)) return false;
  auto strip = [](Value v) {
    while (isa_and_nonnull<arith::ExtFOp, arith::ExtSIOp>(v.getDefiningOp()))
      v = v.getDefiningOp()->getOperand(0);
    return v;
  };
  Value lhs = strip(mul->getOperand(0)), rhs = strip(mul->getOperand(1));
  return (lhs == body->getArgument(0) && rhs == body->getArgument(1)) ||
         (lhs == body->getArgument(1) && rhs == body->getArgument(0));
}

}  // namespace

bool isZvtReadableInputs(Operation *op, bool strictLayoutAndSingleUse) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp || linalgOp.getNumDpsInputs() != 2 ||
      linalgOp.getNumDpsInits() != 1 || !hasZvtMacBody(linalgOp)) {
    return false;
  }
  Value lhs = getZvtLoadSource(linalgOp.getDpsInputOperand(0)->get(),
                               strictLayoutAndSingleUse);
  Value rhs = getZvtLoadSource(linalgOp.getDpsInputOperand(1)->get(),
                               strictLayoutAndSingleUse);
  // The asm checks the element types that it loads, not the types of the
  // operands of `op` (e.g. f32 after an arith.extf of bf16).
  return lhs && rhs &&
         isZvtElementTypes(
             getElementTypeOrSelf(lhs), getElementTypeOrSelf(rhs),
             getElementTypeOrSelf(linalgOp.getDpsInitOperand(0)->get()));
}

// Keep in sync with isZvtMatrixContraction in iree-0006-*.patch
// (looser; only suppresses generic contract lowering),
// isSupportedMatrixContraction in CoralNPUMatrixCodegen.cpp (emits the asm) and
// setZvtVectorSizes in CoralNPUTileSizeSelectionDTCM.cpp (picks the tiles).
bool isZvtMatrixContraction(Operation *op) {
  return hasZvtTargetFeature(op) && isZvtMatrixContractionShape(op);
}

bool isZvtMatrixContractionShape(Operation *op, bool strictLayoutAndSingleUse) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp ||
      (!linalg::isaContractionOpInterface(linalgOp) &&
       !linalg::isaConvolutionOpInterface(linalgOp)) ||
      !isZvtReadableInputs(op, strictLayoutAndSingleUse))
    return false;
  for (Operation *user : linalgOp->getUsers())
    if (auto c = dyn_cast<linalg::LinalgOp>(user);
        c && c.getNumReductionLoops() != 0)
      return false;

  Value init = linalgOp.getDpsInitOperand(0)->get();
  auto initShaped = cast<ShapedType>(init.getType());
  if (!initShaped.hasStaticShape()) return false;

  // The asm zeroes the tiles or seeds them from memory, so any other init
  // (e.g. a broadcast bias) stays on RVV.
  if (auto fill = init.getDefiningOp<linalg::FillOp>())
    init = fill.getInputs()[0];
  if (!matchPattern(init, m_Zero()) && !matchPattern(init, m_AnyZeroFloat()) &&
      !isa_and_nonnull<IREE::Codegen::LoadFromBufferOp,
                       IREE::TensorExt::DispatchTensorLoadOp>(
          init.getDefiningOp()))
    return false;

  if (isa<linalg::Mmt4DOp>(linalgOp)) {
    auto in0Shaped =
        cast<ShapedType>(linalgOp.getDpsInputOperand(0)->get().getType());
    return in0Shaped.hasStaticShape() && initShaped.getDimSize(2) == 16 &&
           initShaped.getDimSize(3) == 16 && in0Shaped.getDimSize(3) == 1;
  }

  if (linalgOp.getNumParallelLoops() < 2 ||
      linalgOp.getNumReductionLoops() < 1) {
    return false;
  }
  // Leading dims must be batch/spatial or outer reduction window offsets.
  SmallVector<unsigned> parallelDims, reductionDims;
  linalgOp.getParallelDims(parallelDims);
  linalgOp.getReductionDims(reductionDims);
  auto dim = [&](unsigned d) { return getAffineDimExpr(d, op->getContext()); };
  AffineExpr mDim = dim(parallelDims[parallelDims.size() - 2]),
             nDim = dim(parallelDims.back()), kDim = dim(reductionDims.back());
  auto isLeadPar = [&](AffineExpr e) {
    auto d = dyn_cast<AffineDimExpr>(e);
    return d && e != mDim && e != nDim &&
           llvm::is_contained(parallelDims, d.getPosition());
  };
  auto isOuterRed = [&](AffineExpr e) {
    auto d = dyn_cast<AffineDimExpr>(e);
    return d && e != kDim && llvm::is_contained(reductionDims, d.getPosition());
  };
  auto isWindow = [&](AffineExpr e, auto pred) {
    auto scaled = [&](AffineExpr x) {
      if (pred(x)) return true;
      auto mul = dyn_cast<AffineBinaryOpExpr>(x);
      return mul && mul.getKind() == AffineExprKind::Mul &&
             pred(mul.getLHS()) && isa<AffineConstantExpr>(mul.getRHS());
    };
    if (scaled(e)) return true;
    auto bin = dyn_cast<AffineBinaryOpExpr>(e);
    return bin && bin.getKind() == AffineExprKind::Add &&
           ((scaled(bin.getLHS()) && isOuterRed(bin.getRHS())) ||
            (isOuterRed(bin.getLHS()) && scaled(bin.getRHS())));
  };
  auto maps = linalgOp.getIndexingMapsArray();
  if (llvm::any_of(maps, [](AffineMap m) { return m.getNumResults() < 2; }) ||
      !llvm::all_of(maps[2].getResults().drop_back(2), isLeadPar) ||
      !llvm::all_of(
          maps[1].getResults().drop_back(2),
          [&](AffineExpr e) { return isLeadPar(e) || isOuterRed(e); }) ||
      !llvm::all_of(maps[0].getResults().drop_back(2),
                    [&](AffineExpr e) { return isWindow(e, isLeadPar); })) {
    return false;
  }
  using Exprs = ArrayRef<AffineExpr>;
  Exprs a = maps[0].getResults().take_back(2),
        b = maps[1].getResults().take_back(2),
        c = maps[2].getResults().take_back(2);
  auto is = [](Exprs t, AffineExpr x, AffineExpr y) {
    return t == Exprs{x, y} || t == Exprs{y, x};
  };
  bool convWindow =
      a[1] == kDim && isWindow(a[0], [&](AffineExpr e) { return e == mDim; });
  if (!is(c, mDim, nDim) ||
      !(((is(a, mDim, kDim) || convWindow) && is(b, kDim, nDim)) ||
        (is(b, mDim, kDim) && is(a, kDim, nDim)))) {
    return false;
  }
  ArrayRef<int64_t> cShape = initShaped.getShape().take_back(2);
  int64_t m = cShape[0], n = cShape[1], k = 1;
  for (unsigned r : reductionDims)
    k = std::max(k, linalgOp.getStaticLoopRanges()[r]);
  bool subTileM = llvm::any_of(llvm::seq<int64_t>(3, 16),
                               [&](int64_t t) { return m % t == 0; });
  return m >= 3 &&
         (m % 16 == 0 || is(a, c[0], kDim) ||
          (convWindow && k > 1 && subTileM)) &&
         n >= 8 &&
         (n % 16 == 0 ||
          ((n % 8 == 0 || n % 12 == 0) && c == Exprs{mDim, nDim} &&
           (is(a, mDim, kDim) || convWindow) && subTileM));
}

bool hasContractionReuse(Operation *op) {
  auto linalgOp = dyn_cast_if_present<linalg::LinalgOp>(op);
  if (!linalgOp || !linalg::isaContractionOpInterface(linalgOp)) return false;
  FailureOr<linalg::ContractionDimensions> dims =
      linalg::inferContractionDims(linalgOp);
  if (failed(dims) || dims->k.empty()) return false;
  SmallVector<int64_t> ranges = linalgOp.getStaticLoopRanges();
  // Each operand slice is reused across the opposite parallel dimension only
  // when that dimension has non-zero extent and > 1 (or dynamic) iterations.
  auto hasExtentReuse = [&](ArrayRef<unsigned> loopIndices) {
    return llvm::none_of(loopIndices,
                         [&](unsigned i) { return ranges[i] == 0; }) &&
           llvm::any_of(loopIndices, [&](unsigned i) {
             return ShapedType::isDynamic(ranges[i]) || ranges[i] > 1;
           });
  };
  return hasExtentReuse(dims->m) && hasExtentReuse(dims->n);
}

}  // namespace mlir::coralnpu_compiler
