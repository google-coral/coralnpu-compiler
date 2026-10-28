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
//
//===----------------------------------------------------------------------===//
// CoralNPU RVV Vector Tile Size and Pipeline Selection
//===----------------------------------------------------------------------===//

#include "compiler/Transforms/CoralNPUTileSizeSelectionRVV.h"

// IREE:
#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUTypes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Utils/Utils.h"

// MLIR:
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"

// LLVM:
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/MathExtras.h"

// Standard library:
#include <algorithm>
#include <iterator>
#include <numeric>
#include <optional>

using namespace mlir;
using namespace mlir::iree_compiler;

namespace mlir::coralnpu_compiler {
namespace {

// Returns true if `op` is a transcendental math op whose polynomial lowering
// needs extra temporary vector registers.
bool isTranscendentalOp(Operation *op) {
  return isa<math::AtanOp, math::Atan2Op, math::CosOp, math::ExpOp,
             math::Exp2Op, math::ExpM1Op, math::LogOp, math::Log10Op,
             math::Log1pOp, math::Log2Op, math::PowFOp, math::SinOp,
             math::TanOp, math::TanhOp, math::ErfOp>(op);
}

// Returns true if `op`'s enclosing block contains any transcendental math op
// (including inside fused consumer `linalg.generic` bodies).
bool hasTranscendentalInBlock(Operation *op) {
  Block *block = op->getBlock();
  if (!block) return false;
  return llvm::any_of(*block, [](Operation &blockOp) {
    return blockOp
        .walk([](Operation *nested) {
          return isTranscendentalOp(nested) ? WalkResult::interrupt()
                                            : WalkResult::advance();
        })
        .wasInterrupted();
  });
}

// Returns true if `op`'s body contains division, remainder, rounding, sqrt, or
// transcendental ops that increase vector register pressure.
bool hasExpensiveBody(linalg::LinalgOp op) {
  Block *body = op.getBlock();
  if (!body) return false;
  return llvm::any_of(body->without_terminator(), [](Operation &bodyOp) {
    return isTranscendentalOp(&bodyOp) ||
           isa<arith::DivFOp, arith::DivSIOp, arith::DivUIOp,
               arith::FloorDivSIOp, arith::CeilDivSIOp, arith::CeilDivUIOp,
               arith::RemFOp, arith::RemSIOp, arith::RemUIOp, math::CeilOp,
               math::FloorOp, math::RoundOp, math::RoundEvenOp, math::SqrtOp,
               math::RsqrtOp, math::IPowIOp>(&bodyOp);
  });
}

// Returns an upper bound on peak simultaneously live vector values in `op`'s
// body, counting block arguments used in the body and adding 2 temporary
// registers during each transcendental op.
int64_t peakLiveValuesInBlock(linalg::LinalgOp op) {
  Block *body = op.getBlock();
  if (!body) return 1;
  llvm::DenseMap<Value, unsigned> lastStep;
  unsigned step = 0;
  for (Operation &bodyOp : *body) {
    ++step;
    for (Value operand : bodyOp.getOperands()) lastStep[operand] = step;
  }
  llvm::DenseSet<Value> live;
  for (BlockArgument arg : body->getArguments()) {
    if (!arg.use_empty()) live.insert(arg);
  }
  int64_t peak = std::max<int64_t>(1, live.size());
  step = 0;
  for (Operation &bodyOp : *body) {
    ++step;
    int64_t extra = isTranscendentalOp(&bodyOp) ? 2 : 0;
    peak = std::max<int64_t>(peak, static_cast<int64_t>(live.size()) + extra);
    for (Value res : bodyOp.getResults()) {
      if (!res.use_empty()) live.insert(res);
    }
    peak = std::max<int64_t>(peak, static_cast<int64_t>(live.size()) + extra);
    for (Value operand : bodyOp.getOperands()) {
      if (lastStep.lookup(operand) == step) live.erase(operand);
    }
  }
  return peak;
}

// Selects `8 * vectorWidth` (`LMUL=8`) when the innermost parallel range is a
// multiple of `8 * vectorWidth`, live values fit in `3/4` of 32 vector
// registers (`<= 24`), no operand has a wider element type than the result,
// and `innerPar` is innermost in every map that uses it; otherwise `4 *
// vectorWidth` (`LMUL=4`).
int64_t getElementwiseTileSize(linalg::LinalgOp op, int64_t innerRange,
                               int64_t vectorWidth, unsigned innerPar) {
  int64_t lmul8 = 8 * vectorWidth;
  if (innerRange < lmul8 || innerRange % lmul8 != 0) return 4 * vectorWidth;
  if (peakLiveValuesInBlock(op) * 8 > 24) return 4 * vectorWidth;
  unsigned outBits =
      getElementTypeOrSelf(op->getResult(0).getType()).getIntOrFloatBitWidth();
  for (Value operand : op->getOperands()) {
    Type elemTy = getElementTypeOrSelf(operand.getType());
    if (elemTy.isIntOrFloat() && elemTy.getIntOrFloatBitWidth() > outBits) {
      return 4 * vectorWidth;
    }
  }
  auto dimExpr = getAffineDimExpr(innerPar, op.getContext());
  for (AffineMap map : op.getIndexingMapsArray()) {
    if (map.isFunctionOfDim(innerPar) &&
        (map.getNumResults() == 0 || map.getResults().back() != dimExpr)) {
      return 4 * vectorWidth;
    }
  }
  return lmul8;
}

// Returns true if `op` is a pure broadcast without arithmetic in its body.
bool isPureBroadcastOp(linalg::LinalgOp op) {
  if (isa<linalg::BroadcastOp>(op.getOperation())) return true;
  auto genericOp = dyn_cast<linalg::GenericOp>(op.getOperation());
  if (!genericOp || genericOp.getNumDpsInputs() != 1 ||
      genericOp.getNumDpsInits() != 1 || !op.hasOnlyProjectedPermutations()) {
    return false;
  }
  Block *body = genericOp.getBlock();
  if (!body || !body->without_terminator().empty()) return false;
  AffineMap inMap = genericOp.getIndexingMapsArray()[0];
  AffineMap outMap = genericOp.getIndexingMapsArray()[1];
  return inMap.getNumResults() < outMap.getNumResults() &&
         outMap.isPermutation();
}

// Returns true if any DPS input indexing map of `op` drops at least one loop
// dimension present in the output map.
bool hasBroadcastInput(linalg::LinalgOp op) {
  if (op.getNumDpsInits() == 0) return false;
  unsigned outRank =
      op.getMatchingIndexingMap(op.getDpsInitOperand(0)).getNumResults();
  return llvm::any_of(op.getDpsInputOperands(), [&](OpOperand *input) {
    return op.getMatchingIndexingMap(input).getNumResults() < outRank;
  });
}

// Returns the largest divisor of `range` in `[1, maxTile]`.
int64_t pickTileDivisor(int64_t range, int64_t maxTile) {
  if (range <= 0) return maxTile;
  for (int64_t t = std::min(range, maxTile); t >= 1; --t) {
    if (range % t == 0) return t;
  }
  return 1;
}

// Returns true if `op` has projected-permutation maps where the innermost
// reduction loop is the last dimension of both the first two input maps (and
// `op` is not a Zvt matrix contraction).
bool isKInnermostInBothInputs(linalg::LinalgOp op,
                              ArrayRef<unsigned> reductionLoops) {
  if (reductionLoops.empty() || op.getNumDpsInputs() < 2 ||
      !op.hasOnlyProjectedPermutations() ||
      isZvtMatrixContraction(op.getOperation())) {
    return false;
  }
  auto kExpr = getAffineDimExpr(reductionLoops.back(), op.getContext());
  SmallVector<AffineMap> maps = op.getIndexingMapsArray();
  AffineMap lhsMap = maps[0];
  AffineMap rhsMap = maps[1];
  return lhsMap.getNumResults() > 0 && rhsMap.getNumResults() > 0 &&
         lhsMap.getResults().back() == kExpr &&
         rhsMap.getResults().back() == kExpr;
}

// Infers contraction dimensions `{batch, m, n, k}` for named or generic
// contractions (including matvec/vecmat where M or N is empty, and 3/4-input
// quantized contractions). Requires either `isaContractionOpInterface` or both
// a multiply and an add in the body so 2-input row reductions are not matched.
FailureOr<linalg::ContractionDimensions> inferContractionLikeDims(
    linalg::LinalgOp linalgOp) {
  if (linalgOp.getNumDpsInputs() < 2 || linalgOp.getNumDpsInits() != 1 ||
      linalgOp.getNumReductionLoops() == 0 ||
      !linalgOp.hasOnlyProjectedPermutations()) {
    return failure();
  }
  bool isNamedContraction = linalg::isaContractionOpInterface(linalgOp);
  if (!isNamedContraction) {
    Block *body = linalgOp.getBlock();
    if (!body) return failure();
    bool hasMul = false;
    bool hasAdd = false;
    for (Operation &bodyOp : body->without_terminator()) {
      hasMul |= isa<arith::MulFOp, arith::MulIOp>(&bodyOp);
      hasAdd |= isa<arith::AddFOp, arith::AddIOp>(&bodyOp);
    }
    if (!hasMul || !hasAdd) return failure();
  }
  SmallVector<AffineMap> maps = linalgOp.getIndexingMapsArray();
  auto dims = linalg::inferContractionDims(
      ArrayRef<AffineMap>{maps[0], maps[1], maps.back()});
  if (failed(dims) || dims->k.empty() ||
      (dims->m.empty() && dims->n.empty() && !isNamedContraction)) {
    return failure();
  }
  return dims;
}

// Sets vector tile sizes `[..., owTile, cTile]` for a channel-last 1D/2D
// convolution or pooling with innermost channel `innerPar` and innermost
// spatial width `secondPar`.
void setChannelLastWindowVectorSizes(ArrayRef<int64_t> loopRanges,
                                     unsigned innerPar, unsigned secondPar,
                                     int64_t vectorWidth, bool unitStrideW,
                                     bool capToLmul4,
                                     MutableArrayRef<int64_t> vectorParallel) {
  int64_t cRange = loopRanges[innerPar];
  int64_t owRange = loopRanges[secondPar];
  int64_t lmul4 = 4 * vectorWidth;
  int64_t lmul8 = 8 * vectorWidth;
  bool useWideOw = false;
  // Unroll 7 output columns at LMUL=4 when `ow` has an odd divisor in {5, 7}
  // (e.g. 5x5, 7x7, 25x25, 35x35 feature maps) and no divisor in {2, 3}. When
  // `owRange` is 5 or 7, `capVectorSizes` turns `7 >= owRange` into `0` so the
  // decomposed 1D slice unrolls all 5 or 7 columns at LMUL=4.
  if (unitStrideW && cRange % lmul4 == 0 && owRange >= 5) {
    bool hasDiv2To3 = owRange % 2 == 0 || owRange % 3 == 0;
    bool hasDiv5Or7 = owRange % 5 == 0 || owRange % 7 == 0;
    useWideOw = hasDiv5Or7 && !hasDiv2To3;
  }
  vectorParallel[innerPar] = (useWideOw || capToLmul4) ? lmul4 : lmul8;
  vectorParallel[secondPar] =
      cRange % vectorWidth != 0 ? 1 : (useWideOw ? 7 : 4);
}

// Selects RVV vector tile sizes and pipeline for convolutions, depthwise
// convolutions, and poolings. Returns true if `linalgOp` matched.
bool matchConvOrPooling(linalg::LinalgOp linalgOp, ArrayRef<int64_t> loopRanges,
                        ArrayRef<unsigned> parallelLoops,
                        ArrayRef<unsigned> reductionLoops, int64_t vectorWidth,
                        IREE::CPU::LoweringPipeline &pipeline,
                        MutableArrayRef<int64_t> vectorParallel,
                        MutableArrayRef<int64_t> vectorReduction) {
  if (parallelLoops.empty() || reductionLoops.empty()) return false;
  auto dims = inferConvDims(linalgOp.getOperation());
  if (failed(dims)) return false;

  unsigned innerPar = parallelLoops.back();
  std::optional<unsigned> secondPar =
      parallelLoops.size() >= 2
          ? std::optional<unsigned>(parallelLoops[parallelLoops.size() - 2])
          : std::nullopt;
  unsigned innerRed = reductionLoops.back();
  unsigned nSpatial = dims->outputImage.size();
  bool isChannelLast = !llvm::is_contained(dims->outputImage, innerPar);
  bool innerAligned = loopRanges[innerPar] % vectorWidth == 0;
  bool isWidening = hasWideningInput(linalgOp);

  // Grouped convolutions have both input/output channel and group (depth) dims.
  if (!dims->inputChannel.empty() && !dims->outputChannel.empty() &&
      !dims->depth.empty()) {
    if (isChannelLast) {
      pipeline = IREE::CPU::LoweringPipeline::ConvTileAndDecomposeExpert;
    }
    vectorParallel[innerPar] = 8 * vectorWidth;
    return true;
  }

  // Depthwise convolutions have a depth dim and no inputChannel reduction dim.
  if (dims->inputChannel.empty() && !dims->depth.empty()) {
    // 2D channel-last depthwise conv without a channel multiplier: size the
    // channel tile and output-width unrolling to fit the 32-register RVV file.
    if (dims->outputChannel.empty() && isChannelLast && nSpatial == 2) {
      pipeline = IREE::CPU::LoweringPipeline::ConvTileAndDecomposeExpert;
      int64_t cRange = loopRanges[innerPar];
      int64_t maxCTile = (isWidening ? 8 : 4) * vectorWidth;
      int64_t cTile = pickTileDivisor(cRange, maxCTile);
      vectorParallel[innerPar] = cTile;
      if (secondPar) {
        int64_t accRegs = 32 / (isWidening ? 2 : 4);
        int64_t cRegs =
            llvm::PowerOf2Ceil(llvm::divideCeil(cTile, vectorWidth));
        vectorParallel[*secondPar] = std::min<int64_t>(accRegs / cRegs, 8);
      }
      if (isWidening && loopRanges[innerRed] <= 5) {
        vectorReduction[innerRed] = loopRanges[innerRed];
      }
      return true;
    }

    if (dims->outputChannel.empty() && (isChannelLast || nSpatial == 1)) {
      pipeline = IREE::CPU::LoweringPipeline::ConvTileAndDecomposeExpert;
    }
    vectorParallel[innerPar] = 4 * vectorWidth;
    if (secondPar &&
        (!dims->outputChannel.empty() || isChannelLast || nSpatial == 1)) {
      vectorParallel[*secondPar] =
          nSpatial >= 3 ? 2 * vectorWidth : (innerAligned ? 2 : 1);
    }
    return true;
  }

  // Poolings have no channel or depth dimensions in ConvolutionDimensions.
  if (dims->inputChannel.empty() && dims->outputChannel.empty() &&
      dims->depth.empty()) {
    if (!isChannelLast) {
      vectorParallel[innerPar] = 4 * vectorWidth;
      return true;
    }
    pipeline = IREE::CPU::LoweringPipeline::ConvTileAndDecomposeExpert;
    if (nSpatial <= 2 && secondPar) {
      setChannelLastWindowVectorSizes(loopRanges, innerPar, *secondPar,
                                      vectorWidth, /*unitStrideW=*/true,
                                      /*capToLmul4=*/false, vectorParallel);
    } else {
      vectorParallel[innerPar] = vectorWidth;
      if (secondPar) vectorParallel[*secondPar] = vectorWidth;
    }
    return true;
  }

  // Regular convolutions: check whether the filter's last dimension is the
  // innermost output channel.
  AffineMap filterMap = linalgOp.getIndexingMapsArray()[1];
  bool isFilterChannelLast =
      !dims->outputChannel.empty() && filterMap.getNumResults() > 0 &&
      filterMap.getResults().back() ==
          getAffineDimExpr(dims->outputChannel.back(), linalgOp.getContext());
  if (isChannelLast && isFilterChannelLast) {
    pipeline = IREE::CPU::LoweringPipeline::ConvTileAndDecomposeExpert;
    if (nSpatial <= 2 && secondPar) {
      bool unitStrideW = !dims->strides.empty() && dims->strides.back() == 1;
      bool capToLmul4 = hasTranscendentalInBlock(linalgOp.getOperation());
      setChannelLastWindowVectorSizes(loopRanges, innerPar, *secondPar,
                                      vectorWidth, unitStrideW, capToLmul4,
                                      vectorParallel);
    } else {
      vectorParallel[innerPar] = 4 * vectorWidth;
      if (secondPar) vectorParallel[*secondPar] = innerAligned ? 4 : 1;
    }
    return true;
  }

  vectorParallel[innerPar] =
      (!isChannelLast && nSpatial == 1 ? 1 : 4) * vectorWidth;
  if (secondPar && (isChannelLast || nSpatial != 2)) {
    vectorParallel[*secondPar] = innerAligned ? 4 : 1;
  }
  return true;
}

// Selects RVV vector tile sizes for contractions (vecmat, matvec, or 2D MxN
// contractions not handled by Zvt or the strategy pass). Returns true if
// `linalgOp` matched.
bool matchContraction(linalg::LinalgOp linalgOp, ArrayRef<int64_t> loopRanges,
                      ArrayRef<unsigned> parallelLoops, int64_t vectorWidth,
                      MutableArrayRef<int64_t> vectorParallel) {
  auto dims = inferContractionLikeDims(linalgOp);
  if (failed(dims)) return false;

  Type outTy = getElementTypeOrSelf(linalgOp->getResult(0).getType());
  int64_t vecTile = (outTy.isF32() ? 8 : 4) * vectorWidth;
  auto isUnit = [&](ArrayRef<unsigned> indices) {
    return llvm::all_of(indices,
                        [&](unsigned d) { return loopRanges[d] <= 1; });
  };
  bool unitM = isUnit(dims->m);
  bool unitN = isUnit(dims->n);
  if (unitM && unitN) return true;

  // vecmat (or 1xN matmul): M is empty or all unit; vectorize along N.
  if (unitM) {
    unsigned nLoop = dims->n.back();
    // On widening i8/i16 -> i32 vecmat, use two LMUL=8 accumulator groups when
    // the peeled N remainder fits in a single LMUL=8 group.
    if (!parallelLoops.empty() && nLoop == parallelLoops.back() &&
        outTy.isInteger(32) &&
        getElementTypeOrSelf(linalgOp.getDpsInputOperand(0)->get().getType())
            .isInteger() &&
        hasWideningInput(linalgOp)) {
      int64_t nRange = loopRanges[nLoop];
      int64_t lmul8 = 8 * vectorWidth;
      int64_t twoLmul8 = 16 * vectorWidth;
      vecTile = (nRange % twoLmul8 <= lmul8) ? twoLmul8 : lmul8;
    }
    vectorParallel[nLoop] = vecTile;
    return true;
  }

  // matvec (or Mx1 matmul): N is empty or all unit; vectorize M when M is
  // innermost in the LHS map, otherwise cap to `min(4, vectorWidth)`.
  if (unitN) {
    unsigned mLoop = dims->m.back();
    AffineMap lhsMap = linalgOp.getIndexingMapsArray()[0];
    bool mInnerInLhs = lhsMap.getNumResults() > 0 &&
                       lhsMap.getResults().back() ==
                           getAffineDimExpr(mLoop, linalgOp.getContext());
    vectorParallel[mLoop] =
        mInnerInLhs ? vecTile : std::min<int64_t>(4, vectorWidth);
    return true;
  }

  // 2D MxN contractions that reach `matchContraction`: either 2-input
  // contractions where `isKInnermostInBothInputs` holds (populated before
  // checking `wideMN`), or 3/4-input quantized contractions (e.g.
  // `linalg.quantized_matmul`) not matched by `isaContractionOpInterface`.
  unsigned mLoop = dims->m.back();
  unsigned nLoop = dims->n.back();
  int64_t mTile = pickTileDivisor(loopRanges[mLoop], 8);
  int64_t nTile = (mTile <= 4 ? 4 : 2) * vectorWidth;
  vectorParallel[mLoop] = mTile;
  vectorParallel[nLoop] = nTile;
  return true;
}

// If `linalgOp` is a standalone 2D+ projected-permutation transpose whose
// input and output maps have different innermost dimensions, tiles both
// dimensions to 8 for 8x8 in-register RVV shuffles and returns true.
bool matchTranspose2D(linalg::LinalgOp linalgOp, bool hasNonRootOps,
                      MutableArrayRef<int64_t> vectorParallel) {
  if (hasNonRootOps || !linalgOp.hasOnlyProjectedPermutations() ||
      linalgOp.getNumDpsInits() == 0) {
    return false;
  }
  AffineMap outMap =
      linalgOp.getMatchingIndexingMap(linalgOp.getDpsInitOperand(0));
  if (outMap.getNumResults() < 2) return false;
  auto outDim = dyn_cast<AffineDimExpr>(outMap.getResults().back());
  if (!outDim) return false;
  for (OpOperand *input : linalgOp.getDpsInputOperands()) {
    AffineMap inMap = linalgOp.getMatchingIndexingMap(input);
    if (inMap.getNumResults() < 2) continue;
    auto inDim = dyn_cast<AffineDimExpr>(inMap.getResults().back());
    if (inDim && inDim.getPosition() != outDim.getPosition() &&
        inMap.isFunctionOfDim(outDim.getPosition())) {
      vectorParallel[inDim.getPosition()] = 8;
      vectorParallel[outDim.getPosition()] = 8;
      return true;
    }
  }
  return false;
}

// Returns true if `linalgOp` is a standalone reduction whose first input's
// innermost dimension is a parallel loop (column reduction).
bool isColumnReduction(linalg::LinalgOp linalgOp,
                       ArrayRef<unsigned> parallelLoops,
                       ArrayRef<unsigned> reductionLoops, bool hasNonRootOps) {
  if (parallelLoops.empty() || hasNonRootOps ||
      linalgOp.getNumDpsInputs() == 0 ||
      !linalgOp.hasOnlyProjectedPermutations()) {
    return false;
  }
  AffineMap inMap = linalgOp.getIndexingMapsArray()[0];
  if (inMap.getNumResults() == 0) return false;
  auto inInner = dyn_cast<AffineDimExpr>(inMap.getResults().back());
  return inInner && !llvm::is_contained(reductionLoops, inInner.getPosition());
}

// Selects base RVV vector tile sizes and pipeline for `root` (before divisor
// snapping).
IREE::CPU::LoweringPipeline selectBaseRVVVectorSizes(
    linalg::LinalgOp linalgOp, ArrayRef<int64_t> loopRanges,
    ArrayRef<unsigned> parallelLoops, ArrayRef<unsigned> reductionLoops,
    bool hasNonRootOps, int64_t vectorWidth,
    MutableArrayRef<int64_t> vectorParallel,
    MutableArrayRef<int64_t> vectorReduction) {
  auto pipeline = IREE::CPU::LoweringPipeline::DoubleTilingExpert;
  llvm::fill(vectorParallel, 0);
  llvm::fill(vectorReduction, 0);
  for (unsigned d : parallelLoops) vectorParallel[d] = 1;
  for (unsigned d : reductionLoops) vectorReduction[d] = 1;

  // 1. All-parallel ops (elementwise, broadcast, transpose).
  if (reductionLoops.empty()) {
    if (parallelLoops.empty()) return pipeline;
    unsigned innerPar = parallelLoops.back();
    int64_t range = loopRanges[innerPar];

    if (matchTranspose2D(linalgOp, hasNonRootOps, vectorParallel)) {
      return pipeline;
    }

    if (isPureBroadcastOp(linalgOp)) {
      vectorParallel[innerPar] =
          (range % vectorWidth == 0 ? 8 : 4) * vectorWidth;
      return pipeline;
    }

    vectorParallel[innerPar] =
        (hasExpensiveBody(linalgOp) && linalg::isElementwise(linalgOp))
            ? 4 * vectorWidth
            : getElementwiseTileSize(linalgOp, range, vectorWidth, innerPar);
    return pipeline;
  }

  // 2. Convolutions, depthwise convolutions, and poolings.
  if (matchConvOrPooling(linalgOp, loopRanges, parallelLoops, reductionLoops,
                         vectorWidth, pipeline, vectorParallel,
                         vectorReduction)) {
    return pipeline;
  }

  // 3. Contractions (including K-innermost-in-both-inputs vectorization).
  bool matchedContraction = matchContraction(
      linalgOp, loopRanges, parallelLoops, vectorWidth, vectorParallel);
  if (isKInnermostInBothInputs(linalgOp, reductionLoops)) {
    unsigned kLoop = reductionLoops.back();
    int64_t kRange = loopRanges[kLoop];
    // When K is contiguous in both inputs and has at least 4 vector lanes,
    // keep a wide 2D MxN parallel tile with a 1x K vector tile if M >= 2 and
    // N >= 4; otherwise reset parallel tiles to 1 and vectorize 4x/8x along K.
    if (std::gcd(vectorWidth, kRange) >= 4) {
      auto eff = [&](unsigned d) {
        return std::min(vectorParallel[d], loopRanges[d]);
      };
      bool wideMN = parallelLoops.size() >= 2 &&
                    eff(parallelLoops.back()) >= 4 &&
                    eff(parallelLoops[parallelLoops.size() - 2]) >= 2;
      if (!wideMN) {
        for (unsigned d : parallelLoops) vectorParallel[d] = 1;
      }
      int64_t kMult = wideMN ? 1 : (hasWideningInput(linalgOp) ? 8 : 4);
      vectorReduction[kLoop] = kMult * vectorWidth;
      return pipeline;
    }
  }
  if (matchedContraction) return pipeline;

  // 4. General reductions (single-op and multi-op reductions).
  unsigned innerRed = reductionLoops.back();
  // Column reduction where the first input's innermost dimension is parallel:
  // vectorize across the innermost parallel dimension.
  if (isColumnReduction(linalgOp, parallelLoops, reductionLoops,
                        hasNonRootOps)) {
    unsigned innerPar = parallelLoops.back();
    int64_t range = loopRanges[innerPar];
    vectorParallel[innerPar] = (range % vectorWidth == 0 ? 8 : 4) * vectorWidth;
    return pipeline;
  }

  // Scale the reduction vector tile by the widening ratio (e.g., 4x for
  // i8 -> i32) so narrow input loads pack full vector registers before
  // widening.
  int64_t effVw = vectorWidth;
  if (hasWideningInput(linalgOp)) {
    unsigned inBits =
        getElementTypeOrSelf(linalgOp.getDpsInputOperand(0)->get().getType())
            .getIntOrFloatBitWidth();
    unsigned outBits = getElementTypeOrSelf(linalgOp->getResult(0).getType())
                           .getIntOrFloatBitWidth();
    effVw = vectorWidth * (outBits / inBits);
  }
  vectorReduction[innerRed] = 4 * effVw;
  return pipeline;
}

// Snaps each active vector tile size in `vectorParallel` and `vectorReduction`
// to a divisor of its static loop range (setting `0` when the tile covers the
// full range, and preserving the full tile for peeling on single-reduction ops
// with one non-unit parallel loop when `gcd(tile, range) < tile / 2 && range %
// tile >= vectorWidth`). Returns false if snapping collapsed a vector-wide
// inner parallel or reduction tile below `min(vectorWidth, 4)`.
bool capVectorSizes(ArrayRef<int64_t> loopRanges,
                    ArrayRef<unsigned> parallelLoops,
                    ArrayRef<unsigned> reductionLoops, int64_t vectorWidth,
                    MutableArrayRef<int64_t> vectorParallel,
                    MutableArrayRef<int64_t> vectorReduction) {
  int64_t minLanes = std::min<int64_t>(vectorWidth, 4);
  unsigned nonUnitPar = llvm::count_if(
      parallelLoops, [&](unsigned d) { return loopRanges[d] > 1; });
  bool canPeelPar = reductionLoops.size() == 1 && nonUnitPar == 1;
  bool collapsed = false;

  auto snapDim = [&](unsigned d, MutableArrayRef<int64_t> sizes, bool isInner,
                     bool allowPeel) {
    int64_t &tile = sizes[d];
    int64_t range = loopRanges[d];
    if (tile <= 0 || range <= 0) return;
    if (tile >= range) {
      tile = 0;
      return;
    }
    int64_t snapped = pickTileDivisor(range, tile);
    // Preserve the full LMUL=8 tile and peel the remainder when divisor
    // snapping would halve the vector width while the remainder is still at
    // least one vector register wide.
    if (allowPeel && std::gcd(tile, range) < tile / 2 &&
        range % tile >= vectorWidth) {
      return;
    }
    if (isInner && tile >= vectorWidth && snapped < minLanes) {
      collapsed = true;
    }
    tile = snapped;
  };

  for (unsigned d : parallelLoops) {
    bool isInner = d == parallelLoops.back();
    snapDim(d, vectorParallel, isInner, /*allowPeel=*/canPeelPar && isInner);
  }
  for (unsigned d : reductionLoops) {
    bool isInner = d == reductionLoops.back();
    snapDim(d, vectorReduction, isInner, /*allowPeel=*/false);
  }
  return !collapsed;
}

// Returns true if `op` is a 9-loop data-tiled convolution `linalg.generic`
// with `[P,P,P,P, R,R,R, P,R]` iterator types and 2D-tiled input/output
// channels.
bool isDataTiledConvGeneric(Operation *op) {
  auto genericOp = dyn_cast<linalg::GenericOp>(op);
  if (!genericOp || genericOp.getNumLoops() != 9 ||
      genericOp.getNumDpsInputs() < 2 || genericOp.getNumDpsInits() != 1) {
    return false;
  }
  auto iters = genericOp.getIteratorTypesArray();
  static constexpr utils::IteratorType kExpected[9] = {
      utils::IteratorType::parallel,  utils::IteratorType::parallel,
      utils::IteratorType::parallel,  utils::IteratorType::parallel,
      utils::IteratorType::reduction, utils::IteratorType::reduction,
      utils::IteratorType::reduction, utils::IteratorType::parallel,
      utils::IteratorType::reduction,
  };
  if (!std::equal(std::begin(kExpected), std::end(kExpected), iters.begin())) {
    return false;
  }
  SmallVector<AffineMap> maps = genericOp.getIndexingMapsArray();
  if (maps[0].getNumResults() != 5 || maps[1].getNumResults() != 6 ||
      maps.back().getNumResults() != 5) {
    return false;
  }
  auto dims = inferConvDims(op);
  return succeeded(dims) && dims->outputChannel.size() == 2 &&
         dims->inputChannel.size() == 2;
}

// Returns true if `root` should keep the RVV vector tile sizes selected by
// `LLVMCPUSelectLoweringStrategyPass`.
bool shouldKeepStrategyRVVSizes(const DTCMRoot &root,
                                IREE::CPU::LoweringPipeline cpuPipeline) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(root.op);
  if (!linalgOp) return true;
  if (isDataTiledConvGeneric(root.op)) return true;
  if (Block *block = root.op->getBlock()) {
    if (llvm::any_of(*block, llvm::IsaPred<linalg::PackOp, linalg::UnPackOp>)) {
      return true;
    }
  }
  if (cpuPipeline != IREE::CPU::LoweringPipeline::Default &&
      cpuPipeline != IREE::CPU::LoweringPipeline::DoubleTilingExpert &&
      cpuPipeline != IREE::CPU::LoweringPipeline::ConvTileAndDecomposeExpert) {
    return true;
  }
  if (hasContractionReuse(root.op) &&
      !isKInnermostInBothInputs(linalgOp, root.reductionLoops)) {
    return true;
  }
  // Keep strategy sizes on bare spatial convolutions (conv_1d/2d/3d without
  // batch or channel dimensions).
  auto convDims = inferConvDims(root.op);
  if (succeeded(convDims) && convDims->batch.empty() &&
      convDims->inputChannel.empty() && convDims->outputChannel.empty() &&
      convDims->depth.empty()) {
    return true;
  }
  return false;
}

// Returns true if `op` has users and every transitive compute user of `op` in
// the same block is (or feeds into) a reduction op.
bool onlyFeedsReduction(Operation *op) {
  if (op->use_empty()) return false;
  Block *block = op->getBlock();
  SmallVector<Operation *> worklist{op};
  llvm::SmallPtrSet<Operation *, 8> visited{op};
  while (!worklist.empty()) {
    Operation *curr = worklist.pop_back_val();
    for (Operation *user : curr->getUsers()) {
      if (user->getBlock() != block) return false;
      if (auto linalgUser = dyn_cast<linalg::LinalgOp>(user)) {
        if (linalgUser.getNumReductionLoops() > 0) continue;
      }
      if (user->use_empty()) return false;
      if (visited.insert(user).second) worklist.push_back(user);
    }
  }
  return true;
}

}  // namespace

std::optional<IREE::CPU::LoweringPipeline> refineRVVTileSizes(
    FunctionOpInterface funcOp, const DTCMRoot &root,
    MutableArrayRef<int64_t> vectorParallelSizes,
    MutableArrayRef<int64_t> vectorReductionSizes) {
  IREE::Codegen::TranslationInfoAttr translationInfo =
      getTranslationInfo(funcOp);
  if (!translationInfo) return std::nullopt;
  auto pipelineAttr = dyn_cast_if_present<IREE::CPU::PipelineAttr>(
      translationInfo.getPassPipeline());
  if (!pipelineAttr ||
      shouldKeepStrategyRVVSizes(root, pipelineAttr.getValue())) {
    return std::nullopt;
  }
  IREE::CPU::LoweringPipeline cpuPipeline = pipelineAttr.getValue();

  auto linalgOp = cast<linalg::LinalgOp>(root.op);
  Type elemType =
      cast<ShapedType>(linalgOp->getResult(0).getType()).getElementType();
  int64_t elemBytes =
      std::max<int64_t>(1, elemType.getIntOrFloatBitWidth() / 8);
  int64_t vlenBits = getVlenFromTargetFeatures(funcOp).value_or(128);
  int64_t vectorWidth = std::max<int64_t>(1, (vlenBits / 8) / elemBytes);

  bool hasNonRootOps = !root.nonRootOps.empty();
  SmallVector<int64_t> candPar(root.loopRanges.size(), 0);
  SmallVector<int64_t> candRed(root.loopRanges.size(), 0);
  IREE::CPU::LoweringPipeline pipeline = selectBaseRVVVectorSizes(
      linalgOp, root.loopRanges, root.parallelLoops, root.reductionLoops,
      hasNonRootOps, vectorWidth, candPar, candRed);

  // If the strategy pass selected ConvTileAndDecomposeExpert (e.g. NCHW
  // pooling) and our rules did not decompose it, keep the strategy sizes.
  if (cpuPipeline == IREE::CPU::LoweringPipeline::ConvTileAndDecomposeExpert &&
      pipeline != IREE::CPU::LoweringPipeline::ConvTileAndDecomposeExpert) {
    return std::nullopt;
  }

  if (!capVectorSizes(root.loopRanges, root.parallelLoops, root.reductionLoops,
                      vectorWidth, candPar, candRed)) {
    return std::nullopt;
  }

  // On non-transpose single-op all-parallel dispatches with a narrow inner
  // dimension and a broadcast input, unroll the second-innermost parallel loop
  // by up to 8 to increase ILP.
  if (!hasNonRootOps && root.reductionLoops.empty() &&
      root.parallelLoops.size() >= 2 && hasBroadcastInput(linalgOp) &&
      peakLiveValuesInBlock(linalgOp) * 8 <= 24) {
    unsigned innerPar = root.parallelLoops.back();
    unsigned outerPar = root.parallelLoops[root.parallelLoops.size() - 2];
    int64_t effInner =
        candPar[innerPar] > 0 ? candPar[innerPar] : root.loopRanges[innerPar];
    if (effInner <= 2 * vectorWidth && candPar[outerPar] == 1 &&
        root.loopRanges[outerPar] > 1) {
      int64_t unroll = pickTileDivisor(root.loopRanges[outerPar], 8);
      candPar[outerPar] = unroll >= root.loopRanges[outerPar] ? 0 : unroll;
    }
  }

  // Synchronize the shared innermost reduction dimension across multi-op
  // reduction dispatches by clamping to the minimum positive tile size across
  // `root.op` and `root.nonRootOps`.
  if (hasNonRootOps && !root.reductionLoops.empty()) {
    unsigned sharedDim = root.reductionLoops.back();
    int64_t sharedRange = root.loopRanges[sharedDim];
    int64_t minTile = candRed[sharedDim] > 0 ? candRed[sharedDim] : sharedRange;
    for (TilingInterface nonRootOp : root.nonRootOps) {
      auto nonRootLinalg = dyn_cast<linalg::LinalgOp>(nonRootOp.getOperation());
      if (!nonRootLinalg || nonRootLinalg->getNumResults() == 0) continue;
      Type nrElemTy =
          getElementTypeOrSelf(nonRootLinalg->getResult(0).getType());
      if (!nrElemTy.isIntOrFloat() || nrElemTy.getIntOrFloatBitWidth() < 8) {
        continue;
      }
      int64_t nrBytes =
          std::max<int64_t>(1, nrElemTy.getIntOrFloatBitWidth() / 8);
      int64_t nrVw = std::max<int64_t>(1, (vlenBits / 8) / nrBytes);
      SmallVector<unsigned> nrPar;
      SmallVector<unsigned> nrRed;
      classifyLoops(nonRootOp, nrPar, nrRed);
      SmallVector<int64_t> nrCandPar(root.loopRanges.size(), 0);
      SmallVector<int64_t> nrCandRed(root.loopRanges.size(), 0);
      selectBaseRVVVectorSizes(nonRootLinalg, root.loopRanges, nrPar, nrRed,
                               /*hasNonRootOps=*/true, nrVw, nrCandPar,
                               nrCandRed);
      if (!capVectorSizes(root.loopRanges, nrPar, nrRed, nrVw, nrCandPar,
                          nrCandRed)) {
        return std::nullopt;
      }
      int64_t opTile =
          nrRed.empty() ? nrCandPar[sharedDim] : nrCandRed[sharedDim];
      if (opTile > 0) minTile = std::min(minTile, opTile);
    }
    candRed[sharedDim] = minTile >= sharedRange ? 0 : minTile;
  }

  llvm::copy(candPar, vectorParallelSizes.begin());
  llvm::copy(candRed, vectorReductionSizes.begin());
  return pipeline;
}

void syncNonRootRVVConfigs(const DTCMRoot &root,
                           ArrayRef<int64_t> vectorParallelSizes,
                           ArrayRef<int64_t> vectorReductionSizes) {
  if (root.nonRootOps.empty() || root.reductionLoops.empty()) return;
  MLIRContext *ctx = root.op->getContext();
  for (TilingInterface nonRootOp : root.nonRootOps) {
    Operation *op = nonRootOp.getOperation();
    SmallVector<unsigned> nrPar;
    SmallVector<unsigned> nrRed;
    classifyLoops(nonRootOp, nrPar, nrRed);
    if (nrRed.empty() && onlyFeedsReduction(op)) continue;

    NamedAttrList items;
    if (llvm::any_of(vectorParallelSizes, [](int64_t s) { return s != 0; })) {
      items.set(IREE::CPU::getTilingLevelName(
                    IREE::CPU::TilingLevel::VectorCommonParallelTiles),
                getTilingLevelAttr(ctx, vectorParallelSizes));
    }
    if (!nrRed.empty()) {
      if (llvm::any_of(vectorReductionSizes,
                       [](int64_t s) { return s != 0; })) {
        items.set(IREE::CPU::getTilingLevelName(
                      IREE::CPU::TilingLevel::VectorReductionTiles),
                  getTilingLevelAttr(ctx, vectorReductionSizes));
      }
    } else if (llvm::any_of(vectorReductionSizes,
                            [](int64_t s) { return s != 0; })) {
      items.set(IREE::CPU::getTilingLevelName(
                    IREE::CPU::TilingLevel::VectorInnerParallelTiles),
                getTilingLevelAttr(ctx, vectorReductionSizes));
    }
    if (!items.empty()) {
      setLoweringConfig(op, IREE::CPU::LoweringConfigAttr::get(
                                ctx, items.getDictionary(ctx)));
    }
  }
}

}  // namespace mlir::coralnpu_compiler
