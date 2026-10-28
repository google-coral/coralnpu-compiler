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
// CoralNPU DTCM Tile Size Selection Pass
//===----------------------------------------------------------------------===//

#include "compiler/Transforms/CoralNPUTileSizeSelectionFootprint.h"
#include "compiler/Transforms/CoralNPUTileSizeSelectionUtils.h"
#include "compiler/Transforms/Passes.h"

// IREE:
#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUDialect.h"
#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUTypes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Utils/CPUUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"

// MLIR:
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Remarks.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Pass/Pass.h"

// LLVM:
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/InterleavedRange.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/MathExtras.h"

// Standard library:
#include <algorithm>
#include <memory>
#include <numeric>
#include <optional>
#include <tuple>
#include <utility>

#define DEBUG_TYPE "coralnpu-tile-size-selection"

using namespace mlir;
using namespace mlir::iree_compiler;

namespace mlir::coralnpu_compiler {

#define GEN_PASS_DEF_CORALNPUTILESIZESELECTIONDTCM
#include "compiler/Transforms/Passes.h.inc"

namespace {

// The root op of a dispatch and its loops, for the DTCM tile-size selection.
struct DTCMRoot {
  Operation *op = nullptr;  // has the strategy config
  IREE::CPU::LoweringConfigAttr config;
  SmallVector<int64_t> loopRanges;          // static ranges
  SmallVector<unsigned> parallelLoops;      // outermost first
  SmallVector<unsigned> reductionLoops;     // outermost first
  SmallVector<TilingInterface> nonRootOps;  // multi-op reduction peers
};

// Appends the parallel and reduction loop indices of `tilingOp` in
// outermost-first order.
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

// Returns the static loop ranges of `op`, or `std::nullopt` if any loop has a
// dynamic range.
std::optional<SmallVector<int64_t>> getStaticLoopRanges(TilingInterface op) {
  OpBuilder builder(op);
  auto [offsets, sizes, strides] =
      getOffsetsSizesAndStrides(op.getIterationDomain(builder));
  return getConstantIntValues(sizes);
}

// Returns true if `map` is a projected permutation whose results are
// `AffineDimExpr`s in strictly increasing dimension order (no constant zeros or
// transposes).
bool isMonotonicProjectedPermutation(AffineMap map) {
  int64_t prevDim = -1;
  for (AffineExpr expr : map.getResults()) {
    auto dimExpr = dyn_cast<AffineDimExpr>(expr);
    if (!dimExpr || static_cast<int64_t>(dimExpr.getPosition()) <= prevDim) {
      return false;
    }
    prevDim = dimExpr.getPosition();
  }
  return true;
}

// Returns true if `op` is a single-result non-contraction LinalgOp with
// monotonic projected-permutation indexing maps, so root-anchored parallel
// tiling can fuse it with a reduction root sharing the same iteration domain.
// Contractions are identified via `isaContractionOpInterface` rather than
// `inferContractionDims` on raw indexing maps so 2-input row reductions over 2D
// tensors are not misclassified as contractions.
bool isSupportedMultiOp(Operation *op) {
  if (op->getNumResults() != 1) return false;
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp || linalg::isaContractionOpInterface(linalgOp)) return false;
  return llvm::all_of(linalgOp.getIndexingMapsArray(),
                      isMonotonicProjectedPermutation);
}

// Returns the root op if the DTCM tile-size selection supports the dispatch;
// else emits a missed remark (unless `computeOps` is empty) and returns
// failure.
FailureOr<DTCMRoot> getDTCMRoot(FunctionOpInterface funcOp,
                                ArrayRef<Operation *> computeOps) {
  // A dispatch without compute ops has nothing to tile.
  if (computeOps.empty()) return failure();

  // Find the unique root op, identified by having workgroup (distribution)
  // tile sizes from LLVMCPUSelectLoweringStrategyPass.
  DTCMRoot root;
  for (Operation *op : computeOps) {
    auto config = getLoweringConfig<IREE::CPU::LoweringConfigAttr>(op);
    if (!config || !config.hasWorkgroupTilingLevel()) continue;
    // More than one root op is unsupported.
    if (root.op) {
      root.op = nullptr;
      break;
    }
    root.op = op;
    root.config = config;
  }
  if (!root.op || !getTranslationInfo(funcOp)) {
    LDBG() << "skip " << funcOp.getName()
           << ": no unique root op with CPU distribution tile sizes and "
              "translation info";
    remark::missed(
        funcOp.getLoc(),
        remark::RemarkOpts::name("DTCMTileSizes").category(DEBUG_TYPE))
        << remark::reason(
               "the dispatch has no unique root op with CPU distribution tile "
               "sizes and translation info; the DTCM tile-size selection skips "
               "the dispatch");
    return failure();
  }

  auto rootTilingOp = dyn_cast<TilingInterface>(root.op);
  if (!rootTilingOp) {
    LDBG() << "skip " << root.op->getName()
           << ": root operation is not a TilingInterface";
    remark::missed(
        root.op->getLoc(),
        remark::RemarkOpts::name("DTCMTileSizes").category(DEBUG_TYPE))
        << remark::reason(
               "root operation is not a TilingInterface; skipping tile size "
               "selection");
    return failure();
  }

  // Require at least one ShapedType result with an integer or float element
  // type of at least 8 bits.
  auto resultType = root.op->getNumResults() > 0
                        ? dyn_cast<ShapedType>(root.op->getResult(0).getType())
                        : ShapedType();
  if (!resultType) {
    LDBG() << "skip " << root.op->getName()
           << ": root operation has no ShapedType result";
    remark::missed(
        root.op->getLoc(),
        remark::RemarkOpts::name("DTCMTileSizes").category(DEBUG_TYPE))
        << remark::reason(
               "root operation has no ShapedType result; skipping tile size "
               "selection");
    return failure();
  }
  Type elemType = resultType.getElementType();
  if (!elemType.isIntOrFloat()) {
    LDBG() << "skip " << root.op->getName()
           << ": only integer and float types are supported";
    remark::missed(
        root.op->getLoc(),
        remark::RemarkOpts::name("DTCMTileSizes").category(DEBUG_TYPE))
        << remark::reason("only integer and float types are supported");
    return failure();
  }
  if (elemType.getIntOrFloatBitWidth() < 8) {
    LDBG() << "skip " << root.op->getName()
           << ": sub-byte types (e.g. i1, i4) are not supported";
    remark::missed(
        root.op->getLoc(),
        remark::RemarkOpts::name("DTCMTileSizes").category(DEBUG_TYPE))
        << remark::reason("sub-byte types (e.g. i1, i4) are not supported");
    return failure();
  }

  // Require static loop ranges.
  std::optional<SmallVector<int64_t>> loopRanges =
      getStaticLoopRanges(rootTilingOp);
  if (!loopRanges) {
    LDBG() << "skip " << root.op->getName()
           << ": dynamic shapes are not supported";
    remark::missed(
        root.op->getLoc(),
        remark::RemarkOpts::name("DTCMTileSizes").category(DEBUG_TYPE))
        << remark::reason("dynamic shapes are not supported");
    return failure();
  }
  root.loopRanges = std::move(*loopRanges);

  classifyLoops(rootTilingOp, root.parallelLoops, root.reductionLoops);

  // CoralNPUTileSizeSelectionDTCM only changes the lowering_config of root.op
  // and relies on root-anchored producer/consumer fusion along root.op's
  // parallel loops to tile other compute ops in the dispatch. Non-root compute
  // ops with reduction loops or more parallel loops than root.op (e.g.
  // decomposed linalg.softmax) are supported only when both root.op and the
  // non-root ops are same-block single-result non-contraction ops with
  // projected-permutation maps sharing root.loopRanges and a single innermost
  // reduction loop.
  const bool rootSupportsNonRootOps =
      !root.loopRanges.empty() &&
      root.reductionLoops == ArrayRef<unsigned>{static_cast<unsigned>(
                                 root.loopRanges.size() - 1)} &&
      isSupportedMultiOp(root.op);
  for (Operation *computeOp : computeOps) {
    if (computeOp == root.op) continue;
    auto tilingOp = dyn_cast<TilingInterface>(computeOp);
    if (!tilingOp) continue;
    SmallVector<unsigned> opParallelLoops;
    SmallVector<unsigned> opReductionLoops;
    classifyLoops(tilingOp, opParallelLoops, opReductionLoops);
    if (opReductionLoops.empty() &&
        opParallelLoops.size() <= root.parallelLoops.size()) {
      continue;
    }

    // Accept same-block elementwise or reduction ops whose iteration domain
    // and reduction loop match the root's single-innermost-reduction domain.
    if (rootSupportsNonRootOps &&
        computeOp->getBlock() == root.op->getBlock() &&
        isSupportedMultiOp(computeOp) &&
        getStaticLoopRanges(tilingOp) == root.loopRanges &&
        (opReductionLoops.empty() || opReductionLoops == root.reductionLoops)) {
      root.nonRootOps.push_back(tilingOp);
      continue;
    }

    LDBG() << "skip the tile-size selection: the non-root op "
           << computeOp->getName() << " needs its own tiling config";
    remark::missed(
        root.op->getLoc(),
        remark::RemarkOpts::name("NonRootTiling").category(DEBUG_TYPE))
        << remark::reason(
               "a non-root compute op has reduction loops or more parallel "
               "loops than the root op; the dispatch keeps the lowering "
               "configs of LLVMCPUSelectLoweringStrategyPass");
    return failure();
  }

  return root;
}

// Shrinks the DTCM tile of one loop of `loopIndices` (which lists the
// outermost loop first), the outermost loop first, and returns true. Returns
// false if no loop can shrink. A tile shrinks to about half, aligned down to
// the alignment of its loop in `alignments` (a loop without an alignment does
// not shrink), and to an exact divisor of the loop range in `ranges` if there
// is one (for an unaligned range, the largest one at most half the tile, else
// the largest one below the tile). The loops whose range is a multiple of the
// alignment shrink first. `zvtMLoop` and `zvtNLoop` are the M and N loops of a
// Zvt root, if any; an unaligned Zvt M without such a divisor does not shrink.
bool shrinkLoops(ArrayRef<unsigned> loopIndices, ArrayRef<int64_t> alignments,
                 ArrayRef<int64_t> ranges, std::optional<unsigned> zvtMLoop,
                 std::optional<unsigned> zvtNLoop,
                 MutableArrayRef<int64_t> dtcmTileSizes) {
  // Shrink the aligned loops first: they can shrink to exact divisors.
  for (bool onlyAligned : {true, false}) {
    for (unsigned loopIdx : loopIndices) {
      int64_t align = alignments[loopIdx];
      int64_t range = ranges[loopIdx];
      // Skip the loops without an alignment, and the tiles at the alignment.
      if (align <= 0 || dtcmTileSizes[loopIdx] <= align) continue;
      if (onlyAligned && range % align != 0) continue;
      int64_t next = std::max<int64_t>(
          align, llvm::alignDown(dtcmTileSizes[loopIdx] / 2, align));
      // An exact divisor of the range leaves no partial DTCM tile. An
      // unaligned Zvt N shrinks to multiples of `align` instead, as Zvt cannot
      // lower all column remainders.
      if (range % align == 0) {
        // Find the largest multiple of `align` that divides the range.
        while (next > align && range % next != 0) next -= align;
      } else if (loopIdx != zvtNLoop) {
        // Find the largest divisor of the range that is at least `align`
        // (at most half the tile if possible), if there is one.
        int64_t div = dtcmTileSizes[loopIdx] / 2;
        while (div >= align && range % div != 0) --div;
        if (div < align) {
          div = dtcmTileSizes[loopIdx] - 1;
          while (div >= align && range % div != 0) --div;
        }
        // Keep Zvt M workgroup slices static to avoid stack-buffer fallback.
        if (div < align && loopIdx == zvtMLoop) continue;
        if (div >= align) next = div;
      }
      dtcmTileSizes[loopIdx] = next;
      return true;
    }
  }
  return false;
}

// Clamps each size in `vectorSizes` to the tile size of the same loop in
// `tileSizes` (0 means that the loop is not tiled). Returns true if a size
// changed.
bool clampToTileSizes(ArrayRef<int64_t> tileSizes,
                      MutableArrayRef<int64_t> vectorSizes) {
  bool changed = false;
  for (auto [tileSize, vectorSize] : llvm::zip_equal(tileSizes, vectorSizes)) {
    // Skip loops that are not tiled, and vector tiles that already fit.
    if (tileSize <= 0 || vectorSize <= tileSize) continue;
    vectorSize = tileSize;
    changed = true;
  }
  return changed;
}

// Snaps each parallel loop's vector tile size in `vectorSizes` so that tiling
// by `dtcmTileSizes` and `vectorSizes` yields a static slice: when the DTCM
// tile divides the loop range in `ranges`, keeps 0 or snaps to
// `std::gcd(dtcmSize, vectorSize)`; otherwise sets 1. Returns true if any size
// changed.
bool snapParallelVectorSizesToStaticTiles(
    ArrayRef<unsigned> parallelLoops, ArrayRef<int64_t> ranges,
    ArrayRef<int64_t> dtcmTileSizes, MutableArrayRef<int64_t> vectorSizes) {
  bool changed = false;
  for (unsigned loopIdx : parallelLoops) {
    int64_t range = ranges[loopIdx];
    if (range <= 0) continue;
    int64_t dtcmSize =
        dtcmTileSizes[loopIdx] > 0 ? dtcmTileSizes[loopIdx] : range;
    int64_t &vectorSize = vectorSizes[loopIdx];
    // A partial DTCM parallel tile has dynamic size, so only step 1 produces a
    // static slice; an exact DTCM parallel tile stays static for any divisor of
    // `dtcmSize` (or 0 when untiled).
    int64_t staticSize =
        range % dtcmSize != 0
            ? 1
            : (vectorSize > 0 ? std::gcd(dtcmSize, vectorSize) : 0);
    if (staticSize == vectorSize) continue;
    vectorSize = staticSize;
    changed = true;
  }
  return changed;
}

// Returns the loops of the Zvt accumulator's rows (M) and columns (N).
std::pair<unsigned, unsigned> getZvtMNLoops(linalg::LinalgOp op) {
  auto c = op.getIndexingMapsArray()[2].getResults().take_back(2);
  return {cast<AffineDimExpr>(c[0]).getPosition(),
          cast<AffineDimExpr>(c[1]).getPosition()};
}

// Sets the largest Zvt tile (up to 4 16x16 tiles as 2x2, 1x4 or 4x1) whose
// DTCM tile fits within `budgetBytes` (Zvt codegen cannot lower a split K, so
// reduction vector sizes are set to 1). Returns false, keeping the RVV sizes,
// if none does.
bool setZvtVectorSizes(const DTCMRoot &root,
                       const DTCMFootprintModel &footprintModel,
                       int64_t budgetBytes,
                       MutableArrayRef<int64_t> vectorParallelSizes,
                       MutableArrayRef<int64_t> vectorReductionSizes) {
  const auto &ranges = root.loopRanges;
  auto linalgOp = cast<linalg::LinalgOp>(root.op);
  auto [m, n] = getZvtMNLoops(linalgOp);
  FailureOr<linalg::ConvolutionDimensions> convDims =
      linalg::inferConvolutionDims(linalgOp);
  bool canPeel = root.reductionLoops.size() == 1 ||
                 (succeeded(convDims) && convDims->filterLoop.size() == 1);
  int64_t mTile = ranges[m] % 32 == 0 ? 32 : 16;
  // If M is not a multiple of 16, take an M-divisor tile (3 to 15 rows) if M
  // needs whole tiles, or if it is not more work than 16-row tiles and a
  // peeled remainder.
  if (ranges[m] % mTile != 0) {
    bool needExactM =
        (root.reductionLoops.size() > 1 && (!canPeel || ranges[m] % 16 < 3)) ||
        ranges[n] % 16 != 0;
    int64_t nTile16 =
        ranges[n] % 64 == 0 ? 64 : (ranges[n] % 32 == 0 ? 32 : 16);
    int64_t nTileT = (ranges[n] >= 64 && ranges[n] % 64 <= 16)
                         ? 64
                         : (ranges[n] >= 32 ? 32 : 16);
    int64_t bTiles = 2 * (ranges[n] / 16);
    int64_t work16 =
        llvm::divideCeil(ranges[m], 16) * (ranges[n] / nTile16 + bTiles);
    int64_t nItersT = llvm::divideCeil(ranges[n], nTileT);
    // Take the largest such divisor.
    for (int64_t t = 15; t >= 3; --t) {
      if (ranges[m] % t == 0 &&
          (needExactM || (ranges[m] / t) * (nItersT + bTiles) <= work16)) {
        mTile = t;
        break;
      }
    }
  }
  bool canPeelN = canPeel && ranges[m] % mTile == 0 && ranges[n] % 8 == 0 &&
                  ranges[n] % 32 != 0 && ranges[n] % 32 <= 16;
  // Avoid a >16-col remainder after 64-col tiles (which peels twice), and a
  // 4-col remainder after 16-col tiles (which Zvt cannot lower).
  int64_t nTile =
      mTile <= 16 && (ranges[n] % 64 == 0 ||
                      (canPeelN && ranges[n] >= 64 && ranges[n] % 64 <= 16))
          ? 64
      : (ranges[n] % 32 == 0 || (canPeelN && ranges[n] >= 32))    ? 32
      : ((canPeel && ranges[n] % 16 != 4) || ranges[n] % 16 == 0) ? 16
      : ranges[n] % 12 == 0                                       ? 12
                                                                  : 8;
  if (nTile == 16 && ranges[m] % 64 == 0) mTile = 64;
  // The smallest DTCM tile that the search reaches with these vector tiles: it
  // does not split K (Zvt streams it), it does not shrink the parallel loops
  // without a vector tile, and it keeps M on an exact divisor of its range.
  SmallVector<int64_t> dtcmTileSizes(ranges);
  for (unsigned i : root.parallelLoops) {
    if (vectorParallelSizes[i] > 0) dtcmTileSizes[i] = vectorParallelSizes[i];
  }
  SmallVector<int64_t> zvtSizes(vectorParallelSizes);
  SmallVector<int64_t> zvtReductionSizes(vectorReductionSizes);
  for (unsigned k : root.reductionLoops) zvtReductionSizes[k] = 1;
  // Halve the Zvt tile until its DTCM tile fits: N first down to 32 columns,
  // then M down to 16 rows, then N down to 16 columns.
  while (true) {
    zvtSizes[n] = nTile < ranges[n] ? nTile : 0;
    zvtSizes[m] = mTile < ranges[m] ? mTile : 0;
    dtcmTileSizes[n] = std::min(nTile, ranges[n]);
    // Snap the minimum M tile up to the smallest divisor of ranges[m] so Zvt M
    // workgroup slices stay static.
    int64_t minM = std::min(mTile, ranges[m]);
    while (ranges[m] % minM != 0) ++minM;
    dtcmTileSizes[m] = minM;
    if (footprintModel.estimate(dtcmTileSizes, zvtSizes, zvtReductionSizes,
                                /*streamReduction=*/true) <= budgetBytes) {
      break;
    }
    if (mTile <= 16 && nTile <= 16) return false;
    (nTile > 32 || (mTile <= 16 && nTile == 32) ? nTile : mTile) /= 2;
  }
  vectorParallelSizes[n] = zvtSizes[n];
  vectorParallelSizes[m] = zvtSizes[m];
  llvm::copy(zvtReductionSizes, vectorReductionSizes.begin());
  return true;
}

// Returns the static tile sizes of `level` from `root.config`, or all zeros if
// `level` is not set.
SmallVector<int64_t> getTilingLevelSizes(const DTCMRoot &root,
                                         IREE::CPU::TilingLevel level) {
  auto levelIdx = static_cast<unsigned>(level);
  if (!root.config.hasTilingLevel(levelIdx)) {
    return SmallVector<int64_t>(root.loopRanges.size(), 0);
  }
  SmallVector<int64_t> sizes =
      root.config.getStaticTilingLevelSizes(levelIdx, root.op);
  assert(sizes.size() == root.loopRanges.size());
  return sizes;
}

// Sets the alignment of any loop in `loopIndices` with a non-positive
// alignment to 1, and returns true if any alignment changed.
bool setZeroAlignmentsToOne(ArrayRef<unsigned> loopIndices,
                            MutableArrayRef<int64_t> alignments) {
  bool changed = false;
  for (unsigned loopIdx : loopIndices) {
    if (alignments[loopIdx] <= 0) {
      alignments[loopIdx] = 1;
      changed = true;
    }
  }
  return changed;
}

// The DTCM and vector tile sizes selected for the root op of a dispatch.
struct SelectedTileSizes {
  SmallVector<int64_t> dtcmTileSizes;
  SmallVector<int64_t> vectorParallelSizes;
  SmallVector<int64_t> vectorReductionSizes;
  bool useZvt = false;
  bool parallelClamped = false;
  bool reductionClamped = false;
};

// Selects the DTCM and vector tile sizes for `root` that fit within
// `budgetBytes` (falling back to a 1x1x... parallel tile with a missed remark
// when `streamReduction` prevents splitting the reduction loop), or emits an op
// error and returns failure.
FailureOr<SelectedTileSizes> selectDTCMTileSizes(
    const DTCMRoot &root, const DTCMFootprintModel &footprintModel,
    int64_t budgetBytes) {
  SelectedTileSizes selected;
  selected.vectorParallelSizes = getTilingLevelSizes(
      root, IREE::CPU::TilingLevel::VectorCommonParallelTiles);
  selected.vectorReductionSizes =
      getTilingLevelSizes(root, IREE::CPU::TilingLevel::VectorReductionTiles);

  // A root without promotion (a Zvt root, or any root other than an RVV
  // contraction with reuse) streams its reduction loops one vector step at a
  // time, and the DTCM tile does not split them. A promoted root copies the
  // reduction range of its DTCM tile to DTCM.
  auto linalgOp = dyn_cast<linalg::LinalgOp>(root.op);
  bool hasReuseContraction = hasContractionReuse(root.op);
  bool streamReduction = !hasReuseContraction;

  // The Zvt asm takes vector tiles with a K of 1, and streams K. It must load
  // both inputs with Zvt element types: a root whose input a fused op computes
  // (e.g. a dequantization) stays on RVV, with promotion.
  std::optional<unsigned> zvtMLoop, zvtNLoop;
  if (isZvtMatrixContraction(root.op) && isZvtReadableInputs(root.op) &&
      setZvtVectorSizes(root, footprintModel, budgetBytes,
                        selected.vectorParallelSizes,
                        selected.vectorReductionSizes)) {
    selected.useZvt = true;
    streamReduction = true;
    std::tie(zvtMLoop, zvtNLoop) = getZvtMNLoops(linalgOp);
  }

  // When the search shrinks a DTCM tile, it aligns the tile to a multiple of
  // these alignments, which start as the vector tile sizes. The search can
  // relax the alignments, so it uses copies and keeps the vector tile sizes.
  SmallVector<int64_t> parallelAlignments(selected.vectorParallelSizes);
  SmallVector<int64_t> reductionAlignments(selected.vectorReductionSizes);

  // Start from the full loop ranges, so that a loop that fits stays untiled.
  // Aligning a full loop down would split it for no gain; a split reduction
  // also changes the order of floating-point sums.
  // TODO(search): the search halves the parallel loops down to the vector tile
  // sizes before it splits a reduction loop, so a large matmul gets a thin
  // tile with full K and little reuse in DTCM. Search for the tile with the
  // best reuse for the budget.
  selected.dtcmTileSizes = root.loopRanges;

  bool fallback = false;
  // Shrink the DTCM tile until its buffers (and Zvt scratch tiles, if any) fit
  // in the budget.
  while (footprintModel.estimate(
             selected.dtcmTileSizes, selected.vectorParallelSizes,
             selected.vectorReductionSizes, streamReduction) > budgetBytes) {
    if (shrinkLoops(root.parallelLoops, parallelAlignments, root.loopRanges,
                    zvtMLoop, zvtNLoop, selected.dtcmTileSizes)) {
      continue;
    }

    // The DTCM tile does not split streamed reduction loops (including
    // multi-op reductions, where CacheReductionTiles would recompute producer
    // reductions on every outer reduction step).
    if (!streamReduction &&
        shrinkLoops(root.reductionLoops, reductionAlignments, root.loopRanges,
                    /*zvtMLoop=*/std::nullopt, /*zvtNLoop=*/std::nullopt,
                    selected.dtcmTileSizes)) {
      continue;
    }

    // If any of the align sizes are 0, their dimensions were not shrunken;
    // set these alignments to 1, and try to shrink again.
    bool unalignedFallback =
        setZeroAlignmentsToOne(root.parallelLoops, parallelAlignments);
    if (!streamReduction) {
      unalignedFallback |=
          setZeroAlignmentsToOne(root.reductionLoops, reductionAlignments);
    }
    if (unalignedFallback) continue;

    if (!fallback) {
      // Give up on alignment, set all of them to 1 and try to shrink again.
      LDBG() << root.op->getName()
             << ": workload cannot fit in DTCM with resolved alignments; "
                "falling back to alignment 1";
      remark::missed(
          root.op->getLoc(),
          remark::RemarkOpts::name("DTCMTileSizes").category(DEBUG_TYPE))
          << remark::reason(
                 "workload cannot fit in DTCM with resolved alignments; "
                 "falling back to alignment 1");

      parallelAlignments.assign(parallelAlignments.size(), 1);
      reductionAlignments.assign(reductionAlignments.size(), 1);

      fallback = true;
      continue;
    }

    // Roots with streamed reduction loops do not allocate promoted DTCM
    // buffers and do not split their reduction loops, so keep the 1x1x...
    // parallel tile when a single parallel element exceeds the budget.
    if (streamReduction) {
      LDBG() << root.op->getName()
             << ": workload cannot fit in DTCM even with 1x1x... parallel tile "
                "size; keeping 1x1x... parallel tile";
      remark::missed(
          root.op->getLoc(),
          remark::RemarkOpts::name("DTCMTileSizes").category(DEBUG_TYPE))
          << remark::reason(
                 "workload cannot fit in DTCM even with 1x1x... parallel tile "
                 "size; keeping 1x1x... parallel tile");
      break;
    }

    // Nothing helped, we couldn't find a tile size that fits in DTCM.
    root.op->emitOpError(
        "workload cannot fit in DTCM even with 1x1x... tile size");
    return failure();
  }

  // Set untiled dimensions to 0.
  for (auto [tileSize, range] :
       llvm::zip_equal(selected.dtcmTileSizes, root.loopRanges)) {
    if (tileSize == range) tileSize = 0;
  }

  // A vector tile must fit in the DTCM tile that contains it. The search above
  // can pick a DTCM tile that is smaller than the vector tile when it relaxes
  // the alignments.
  selected.parallelClamped =
      clampToTileSizes(selected.dtcmTileSizes, selected.vectorParallelSizes);
  selected.reductionClamped =
      clampToTileSizes(selected.dtcmTileSizes, selected.vectorReductionSizes);
  // Keep parallel slices reaching VectorReductionTiles static on multi-op
  // reductions so identity extract_slice ops on producer reduction results fold
  // away instead of fusing producer reductions inside the root's reduction
  // loop.
  if (!root.nonRootOps.empty()) {
    selected.parallelClamped |= snapParallelVectorSizesToStaticTiles(
        root.parallelLoops, root.loopRanges, selected.dtcmTileSizes,
        selected.vectorParallelSizes);
  }

  LDBG() << "Selected DTCM tile sizes: "
         << llvm::interleaved_array(selected.dtcmTileSizes);
  LDBG() << "Estimated footprint: "
         << footprintModel.estimate(
                selected.dtcmTileSizes, selected.vectorParallelSizes,
                selected.vectorReductionSizes, streamReduction)
         << " bytes";

  return selected;
}

// Enables loop peeling in the translation info of `funcOp`.
LogicalResult enableLoopPeeling(FunctionOpInterface funcOp) {
  MLIRContext *context = funcOp.getContext();
  IREE::Codegen::TranslationInfoAttr translationInfo =
      getTranslationInfo(funcOp);
  NamedAttrList config(translationInfo.getConfiguration());
  config.set(getEnableLoopPeelingStr(), UnitAttr::get(context));
  translationInfo = IREE::Codegen::TranslationInfoAttr::get(
      context, translationInfo.getPassPipeline(),
      translationInfo.getCodegenSpec(), translationInfo.getWorkgroupSize(),
      translationInfo.getSubgroupSize(), config.getDictionary(context));
  return setTranslationInfo(funcOp, translationInfo);
}

// Sets the tiling level `level` to `sizes` in `configItems`.
void setTilingLevel(MLIRContext *context, IREE::CPU::TilingLevel level,
                    ArrayRef<int64_t> sizes, NamedAttrList &configItems) {
  configItems.set(IREE::CPU::getTilingLevelName(level),
                  getTilingLevelAttr(context, sizes));
}

// Writes `selected` to the lowering config of `root.op`, and updates `funcOp`
// and non-root `computeOps` when Zvt is used.
LogicalResult applySelectedTileSizes(FunctionOpInterface funcOp,
                                     const DTCMRoot &root,
                                     ArrayRef<Operation *> computeOps,
                                     const SelectedTileSizes &selected) {
  MLIRContext *context = funcOp.getContext();
  int64_t numLoops = root.loopRanges.size();
  SmallVector<int64_t> distParallelSizes(numLoops, 0);
  SmallVector<int64_t> cacheReductionSizes(numLoops, 0);
  for (unsigned parallelLoopIdx : root.parallelLoops) {
    distParallelSizes[parallelLoopIdx] =
        selected.dtcmTileSizes[parallelLoopIdx];
  }
  for (unsigned reductionLoopIdx : root.reductionLoops) {
    cacheReductionSizes[reductionLoopIdx] =
        selected.dtcmTileSizes[reductionLoopIdx];
  }

  NamedAttrList configItems(root.config.getConfig());
  // Each workgroup computes one DTCM parallel tile.
  setTilingLevel(context, IREE::CPU::TilingLevel::DistributionTiles,
                 distParallelSizes, configItems);
  setTilingLevel(context, IREE::CPU::TilingLevel::CacheParallelTiles,
                 distParallelSizes, configItems);
  setTilingLevel(context, IREE::CPU::TilingLevel::CacheReductionTiles,
                 cacheReductionSizes, configItems);

  // Write back the vector tile sizes only if this pass changed them (Zvt sizes
  // or clamped sizes), so that the other attributes of the levels stay.
  if (selected.useZvt || selected.parallelClamped) {
    setTilingLevel(context, IREE::CPU::TilingLevel::VectorCommonParallelTiles,
                   selected.vectorParallelSizes, configItems);
  }
  if (selected.useZvt || selected.reductionClamped) {
    setTilingLevel(context, IREE::CPU::TilingLevel::VectorReductionTiles,
                   selected.vectorReductionSizes, configItems);
  }
  // Keep non-root ops' vector tile sizes synchronized when strategy configs
  // are clamped or snapped to static slices.
  if (!selected.useZvt &&
      (selected.parallelClamped || selected.reductionClamped)) {
    for (TilingInterface nonRootOp : root.nonRootOps) {
      Operation *op = nonRootOp.getOperation();
      auto nrConfig = getLoweringConfig<IREE::CPU::LoweringConfigAttr>(op);
      if (!nrConfig) continue;
      NamedAttrList nrItems(nrConfig.getConfig());
      if (selected.parallelClamped &&
          nrConfig.hasTilingLevel(static_cast<unsigned>(
              IREE::CPU::TilingLevel::VectorCommonParallelTiles))) {
        setTilingLevel(context,
                       IREE::CPU::TilingLevel::VectorCommonParallelTiles,
                       selected.vectorParallelSizes, nrItems);
      }
      if (selected.reductionClamped) {
        for (auto level : {IREE::CPU::TilingLevel::VectorReductionTiles,
                           IREE::CPU::TilingLevel::VectorInnerParallelTiles}) {
          if (nrConfig.hasTilingLevel(static_cast<unsigned>(level))) {
            setTilingLevel(context, level, selected.vectorReductionSizes,
                           nrItems);
          }
        }
      }
      setLoweringConfig(op, IREE::CPU::LoweringConfigAttr::get(
                                context, nrItems.getDictionary(context)));
    }
  }

  // Zvt needs loop peeling, and the other compute ops must follow the Zvt
  // tiles of the root op.
  if (selected.useZvt) {
    // Peel partial DTCM and vector tiles (e.g. 80 by 64): Zvt needs static
    // shapes. setTranslationInfo cannot fail today; propagate a failure anyway.
    if (failed(enableLoopPeeling(funcOp))) return failure();

    // The strategy pass derived the vector tile sizes of the other compute ops
    // from the RVV tile sizes of the root op. Erase their lowering configs, so
    // that these ops follow the Zvt tiles of the root op through fusion.
    for (Operation *computeOp : computeOps) {
      // Keep the lowering config of the root op.
      if (computeOp == root.op) continue;
      eraseLoweringConfig(computeOp);
    }
  }

  // Ask CoralNPUPromoteOperandsPass to copy the tiles of all operands of an
  // RVV contraction with reuse to DTCM (the inputs, the accumulator and the
  // output).
  // TODO(zvt): promote the operands of Zvt contractions too.
  if (!selected.useZvt && hasContractionReuse(root.op)) {
    SmallVector<int64_t> operandNumbers =
        llvm::to_vector(llvm::seq<int64_t>(0, root.op->getNumOperands()));
    configItems.set(kPromoteOperandsName,
                    Builder(context).getI64ArrayAttr(operandNumbers));
  }

  setLoweringConfig(root.op, IREE::CPU::LoweringConfigAttr::get(
                                 context, configItems.getDictionary(context)));
  return success();
}

struct CoralNPUTileSizeSelectionDTCMPass
    : public impl::CoralNPUTileSizeSelectionDTCMBase<
          CoralNPUTileSizeSelectionDTCMPass> {
  using CoralNPUTileSizeSelectionDTCMBase::CoralNPUTileSizeSelectionDTCMBase;

  void runOnOperation() override {
    auto funcOp = getOperation();
    auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(funcOp);
    if (!targetAttr || targetAttr.getBackend().getValue() != "coralnpu") {
      return;
    }

    // The DTCM space for the buffers of one tile: DTCM minus the stack and the
    // reserve for static data.
    int64_t budgetBytes =
        getDTCMTileBudgetBytes(dtcmSizeKb, stackSizeKb, dtcmReserveKb);

    LDBG() << "Running CoralNPUTileSizeSelectionDTCMPass (DTCM tile budget: "
           << budgetBytes << " bytes) on function: " << funcOp.getName();

    // Invalid pass options: no budget is left, or no input buffer.
    if (budgetBytes <= 0 || buffersPerOperand < 1) {
      funcOp.emitError("invalid DTCM options: dtcm-size-kb=")
          << dtcmSizeKb << ", stack-size-kb=" << stackSizeKb
          << ", dtcm-reserve-kb=" << dtcmReserveKb
          << " leave a DTCM tile budget of " << budgetBytes
          << " bytes, and buffers-per-operand=" << buffersPerOperand
          << "; the budget must be positive and buffers-per-operand at least 1";
      signalPassFailure();
      return;
    }

    SmallVector<Operation *> computeOps = getComputeOps(funcOp);
    FailureOr<DTCMRoot> root = getDTCMRoot(funcOp, computeOps);
    if (failed(root)) return;

    DTCMFootprintModel footprintModel(root->op, computeOps, buffersPerOperand);
    FailureOr<SelectedTileSizes> selected =
        selectDTCMTileSizes(*root, footprintModel, budgetBytes);
    if (failed(selected)) {
      signalPassFailure();
      return;
    }
    if (failed(applySelectedTileSizes(funcOp, *root, computeOps, *selected))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createCoralNPUTileSizeSelectionDTCMPass() {
  return std::make_unique<CoralNPUTileSizeSelectionDTCMPass>();
}

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createCoralNPUTileSizeSelectionDTCMPass(
    CoralNPUTileSizeSelectionDTCMOptions options) {
  return std::make_unique<CoralNPUTileSizeSelectionDTCMPass>(
      std::move(options));
}

}  // namespace mlir::coralnpu_compiler
