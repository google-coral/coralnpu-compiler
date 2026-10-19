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

#include <algorithm>
#include <memory>
#include <optional>
#include <utility>

#include "compiler/Transforms/CoralNPUTileSizeSelectionAnalysis.h"
#include "compiler/Transforms/CoralNPUTileSizeSelectionFootprint.h"
#include "compiler/Transforms/CoralNPUTileSizeSelectionUtils.h"
#include "compiler/Transforms/Passes.h"

// IREE headers
#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUDialect.h"
#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUTypes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Utils/CPUUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"

// MLIR headers
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Remarks.h"
#include "mlir/Pass/Pass.h"

// LLVM headers
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "coralnpu-tile-size-selection"

using namespace mlir;
using namespace mlir::iree_compiler;

namespace mlir::coralnpu_compiler {

#define GEN_PASS_DEF_CORALNPUTILESIZESELECTIONDTCM
#include "compiler/Transforms/Passes.h.inc"

namespace {

// Shrinks the DTCM tile of one loop of `loopIndices` (which lists the
// innermost loop first), the outermost loop first, and returns true. Returns
// false if no loop can shrink. A tile shrinks to about half, aligned down to
// the alignment of its loop in `alignments` (a loop without an alignment does
// not shrink), and to an exact divisor of the loop range in `ranges` if there
// is one. The loops whose range is a multiple of the alignment shrink first.
// `zvtNLoop` is the N loop of a Zvt root, if any.
bool shrinkLoops(ArrayRef<size_t> loopIndices, ArrayRef<int64_t> alignments,
                 ArrayRef<int64_t> ranges, std::optional<size_t> zvtNLoop,
                 MutableArrayRef<int64_t> dtcmTileSizes) {
  // Shrink the aligned loops first: they can shrink to exact divisors.
  for (bool onlyAligned : {true, false}) {
    for (size_t loopIdx : llvm::reverse(loopIndices)) {
      int64_t align = alignments[loopIdx], range = ranges[loopIdx];
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
        // Find the largest divisor of the range that is at least `align`, if
        // there is one.
        int64_t div = dtcmTileSizes[loopIdx] / 2;
        while (div >= align && range % div != 0) --div;
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

// Returns the loops of the Zvt accumulator's rows (M) and columns (N).
std::pair<size_t, size_t> getZvtMNLoops(linalg::LinalgOp op) {
  auto c = op.getIndexingMapsArray()[2].getResults().take_back(2);
  return {cast<AffineDimExpr>(c[0]).getPosition(),
          cast<AffineDimExpr>(c[1]).getPosition()};
}

// Sets the largest Zvt tile (up to 4 16x16 tiles as 2x2, 1x4 or 4x1) whose
// DTCM tile fits (Zvt codegen cannot lower a split K). Returns false, keeping
// the RVV sizes, if none does. `fits` takes the DTCM tile sizes and the vector
// tile sizes of the parallel loops.
bool setZvtVectorSizes(
    const CoralNPUTileSizeSelectionAnalysis &analysis,
    llvm::function_ref<bool(ArrayRef<int64_t>, ArrayRef<int64_t>)> fits,
    MutableArrayRef<int64_t> vectorParallelSizes) {
  const auto &ranges = analysis.staticLoopRanges;
  auto linalgOp = cast<linalg::LinalgOp>(*analysis.rootTilingOp);
  auto [m, n] = getZvtMNLoops(linalgOp);
  FailureOr<linalg::ConvolutionDimensions> convDims =
      linalg::inferConvolutionDims(linalgOp);
  bool canPeel = analysis.reductionLoops.size() == 1 ||
                 (succeeded(convDims) && convDims->filterLoop.size() == 1);
  int64_t mTile = ranges[m] % 32 == 0 ? 32 : 16;
  // If M is not a multiple of 16, take an M-divisor tile (3 to 15 rows) if M
  // needs whole tiles, or if it is not more work than 16-row tiles and a
  // peeled remainder.
  if (ranges[m] % mTile != 0) {
    bool needExactM = (analysis.reductionLoops.size() > 1 &&
                       (!canPeel || ranges[m] % 16 < 3)) ||
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
  // does not split K (Zvt streams it), and it does not shrink the parallel
  // loops without a vector tile.
  SmallVector<int64_t> dtcmTileSizes(ranges);
  for (size_t i : analysis.parallelLoops) {
    if (vectorParallelSizes[i] > 0) dtcmTileSizes[i] = vectorParallelSizes[i];
  }
  SmallVector<int64_t> zvtSizes(vectorParallelSizes);
  // Halve the Zvt tile until its DTCM tile fits: N first down to 32 columns,
  // then M down to 16 rows, then N down to 16 columns.
  while (true) {
    zvtSizes[n] = nTile < ranges[n] ? nTile : 0;
    zvtSizes[m] = mTile < ranges[m] ? mTile : 0;
    dtcmTileSizes[n] = std::min(nTile, ranges[n]);
    dtcmTileSizes[m] = std::min(mTile, ranges[m]);
    if (fits(dtcmTileSizes, zvtSizes)) break;
    if (mTile <= 16 && nTile <= 16) return false;
    (nTile > 32 || (mTile <= 16 && nTile == 32) ? nTile : mTile) /= 2;
  }
  vectorParallelSizes[n] = zvtSizes[n];
  vectorParallelSizes[m] = zvtSizes[m];
  return true;
}

struct CoralNPUTileSizeSelectionDTCMPass
    : public impl::CoralNPUTileSizeSelectionDTCMBase<
          CoralNPUTileSizeSelectionDTCMPass> {
  using CoralNPUTileSizeSelectionDTCMBase::CoralNPUTileSizeSelectionDTCMBase;

  void runOnOperation() override {
    auto funcOp = getOperation();
    MLIRContext *context = &getContext();

    auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(funcOp);
    if (!targetAttr || targetAttr.getBackend().getValue() != "coralnpu") {
      return;
    }

    // The DTCM space for the buffers of one tile: DTCM minus the stack and the
    // reserve for static data.
    int64_t budgetBytes =
        getDTCMTileBudgetBytes(dtcmSizeKb, stackSizeKb, dtcmReserveKb);

    LLVM_DEBUG(
        llvm::dbgs()
        << "Running CoralNPUTileSizeSelectionDTCMPass (DTCM tile budget: "
        << budgetBytes << " bytes) on function: " << funcOp.getName() << "\n");

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

    auto &analysis = getAnalysis<CoralNPUTileSizeSelectionAnalysis>();
    // The analysis does not support the dispatch. The IR does not change, so
    // the Workgroup pass reuses the analysis.
    if (failed(analysis.status)) {
      markAllAnalysesPreserved();
      return;
    }

    auto tilingOp = analysis.rootTilingOp;

    // LLVMCPUSelectLoweringStrategyPass sets the lowering config of the root op
    // and the translation info of the function; this pass refines them. Skip
    // the dispatch if the root op has no CPU lowering config with distribution
    // tile sizes.
    auto cpuLoweringConfig =
        getLoweringConfig<IREE::CPU::LoweringConfigAttr>(tilingOp);
    IREE::Codegen::TranslationInfoAttr translationInfo =
        getTranslationInfo(funcOp);
    if (!cpuLoweringConfig || !cpuLoweringConfig.hasWorkgroupTilingLevel() ||
        !translationInfo) {
      LDBG() << "skip " << tilingOp->getName()
             << ": no CPU lowering config with distribution tile sizes";
      remark::missed(
          tilingOp->getLoc(),
          remark::RemarkOpts::name("DTCMTileSizes").category(DEBUG_TYPE))
          << remark::reason(
                 "the root op has no CPU lowering config with distribution "
                 "tile sizes; the DTCM tile-size selection skips the dispatch");
      markAllAnalysesPreserved();
      return;
    }

    int64_t numLoops = analysis.staticLoopRanges.size();
    SmallVector<int64_t> vectorParallelSizes(numLoops, 0);
    SmallVector<int64_t> vectorReductionSizes(numLoops, 0);

    auto vectorParallelLevel =
        IREE::CPU::TilingLevel::VectorCommonParallelTiles;
    if (cpuLoweringConfig.hasTilingLevel(
            static_cast<unsigned>(vectorParallelLevel))) {
      vectorParallelSizes = cpuLoweringConfig.getStaticTilingLevelSizes(
          static_cast<unsigned>(vectorParallelLevel), tilingOp);
      assert(vectorParallelSizes.size() == numLoops);
    }

    auto vectorReductionLevel = IREE::CPU::TilingLevel::VectorReductionTiles;
    if (cpuLoweringConfig.hasTilingLevel(
            static_cast<unsigned>(vectorReductionLevel))) {
      vectorReductionSizes = cpuLoweringConfig.getStaticTilingLevelSizes(
          static_cast<unsigned>(vectorReductionLevel), tilingOp);
      assert(vectorReductionSizes.size() == numLoops);
    }

    // The RVV vector tile sizes of the strategy pass, for a root that does not
    // use Zvt.
    const SmallVector<int64_t> rvvParallelSizes(vectorParallelSizes);
    const SmallVector<int64_t> rvvReductionSizes(vectorReductionSizes);

    // A Zvt root and a convolution stream their reduction loops one vector step
    // at a time, and the DTCM tile does not split them. The input slices of the
    // other roots (e.g. an RVV contraction) span the reduction range of the
    // DTCM tile.
    auto linalgOp = dyn_cast<linalg::LinalgOp>(tilingOp.getOperation());
    bool isConvolution =
        linalgOp && linalg::isaConvolutionOpInterface(linalgOp);
    bool streamReduction = isConvolution;

    // A tile fits if the DTCM buffers of the root op and of the ops fused with
    // it, and the scratch tiles of the Zvt asm for the vector tile sizes
    // `parallelSizes` and vectorReductionSizes, fit in the budget.
    DTCMFootprintModel footprintModel(tilingOp, getComputeOps(funcOp),
                                      buffersPerOperand);
    auto fits = [&](ArrayRef<int64_t> tileSizes,
                    ArrayRef<int64_t> parallelSizes) {
      return footprintModel.estimate(tileSizes, parallelSizes,
                                     vectorReductionSizes,
                                     streamReduction) <= budgetBytes;
    };

    // The Zvt asm takes vector tiles with a K of 1, and streams K. It must load
    // both inputs with Zvt element types: a root whose input a fused op
    // computes (e.g. a dequantization) stays on RVV.
    bool useZvt = false;
    if (isZvtMatrixContraction(tilingOp) && isZvtReadableInputs(tilingOp)) {
      for (size_t k : analysis.reductionLoops) vectorReductionSizes[k] = 1;
      streamReduction = true;
      useZvt = setZvtVectorSizes(analysis, fits, vectorParallelSizes);
    }
    if (!useZvt) {
      vectorParallelSizes = rvvParallelSizes;
      vectorReductionSizes = rvvReductionSizes;
      streamReduction = isConvolution;
    }
    std::optional<size_t> zvtNLoop;
    if (useZvt) zvtNLoop = getZvtMNLoops(linalgOp).second;

    // When the search shrinks a DTCM tile, it aligns the tile to a multiple of
    // these alignments, which start as the vector tile sizes. The search can
    // relax the alignments, so it uses copies and keeps the vector tile sizes.
    SmallVector<int64_t> parallelAlignments(vectorParallelSizes);
    SmallVector<int64_t> reductionAlignments(vectorReductionSizes);

    // Start from the full loop ranges, so that a loop that fits stays untiled.
    // Aligning a full loop down would split it for no gain; a split reduction
    // also changes the order of floating-point sums.
    // TODO(search): the search halves the parallel loops down to the vector
    // tile sizes before it splits a reduction loop, so a large matmul gets a
    // thin tile with full K and little reuse in DTCM. Search for the tile with
    // the best reuse for the budget.
    SmallVector<int64_t> dtcmTileSizes(analysis.staticLoopRanges);

    bool fallback = false;
    while (!fits(dtcmTileSizes, vectorParallelSizes)) {
      if (shrinkLoops(analysis.parallelLoops, parallelAlignments,
                      analysis.staticLoopRanges, zvtNLoop, dtcmTileSizes))
        continue;

      // The DTCM tile does not split streamed reduction loops.
      if (!streamReduction &&
          shrinkLoops(analysis.reductionLoops, reductionAlignments,
                      analysis.staticLoopRanges, /*zvtNLoop=*/std::nullopt,
                      dtcmTileSizes))
        continue;

      // If any of the align sizes are 0, their dimensions were not shrunken;
      // set these alignments to 1, and try to shrink again.
      bool unalignedFallback = false;
      for (auto &size : parallelAlignments) {
        if (size <= 0) {
          size = 1;
          unalignedFallback = true;
        }
      }
      for (auto &size : reductionAlignments) {
        if (size <= 0) {
          size = 1;
          unalignedFallback = true;
        }
      }
      if (unalignedFallback) continue;  // try to shrink again

      if (!fallback) {
        // Give up on alignment, set all of them to 1 and try to shrink again.
        tilingOp->emitWarning(
            "workload cannot fit in DTCM with resolved alignments; falling "
            "back to alignment 1");

        parallelAlignments.assign(parallelAlignments.size(), 1);
        reductionAlignments.assign(reductionAlignments.size(), 1);

        fallback = true;
        continue;
      }

      // Nothing helped, we couldn't find a tile size that fits in DTCM.
      tilingOp->emitOpError(
          "workload cannot fit in DTCM even with 1x1x... tile size");
      signalPassFailure();
      return;
    }

    // Set untiled dimensions to 0
    for (size_t i = 0; i < dtcmTileSizes.size(); ++i) {
      dtcmTileSizes[i] = dtcmTileSizes[i] == analysis.staticLoopRanges[i]
                             ? 0
                             : dtcmTileSizes[i];
    }

    // A vector tile must fit in the DTCM tile that contains it. The search
    // above can pick a DTCM tile that is smaller than the vector tile when it
    // relaxes the alignments.
    bool parallelClamped = clampToTileSizes(dtcmTileSizes, vectorParallelSizes);
    bool reductionClamped =
        clampToTileSizes(dtcmTileSizes, vectorReductionSizes);

    LLVM_DEBUG({
      llvm::dbgs() << "    Selected DTCM tile sizes: [";
      for (auto size : dtcmTileSizes) llvm::dbgs() << size << " ";
      llvm::dbgs() << "]\n";
      llvm::dbgs() << "    Estimated footprint: "
                   << footprintModel.estimate(
                          dtcmTileSizes, vectorParallelSizes,
                          vectorReductionSizes, streamReduction)
                   << " bytes\n";
    });

    SmallVector<int64_t> distParallelSizes(numLoops, 0);
    SmallVector<int64_t> cacheReductionSizes(numLoops, 0);

    for (size_t vectorLoopIdx : analysis.parallelLoops) {
      distParallelSizes[vectorLoopIdx] = dtcmTileSizes[vectorLoopIdx];
    }

    for (size_t reductionLoopIdx : analysis.reductionLoops) {
      cacheReductionSizes[reductionLoopIdx] = dtcmTileSizes[reductionLoopIdx];
    }

    SmallVector<NamedAttribute> configItems(
        cpuLoweringConfig.getConfig().getValue());

    // Sets the config entry `name` to `attr`.
    auto setConfigItem = [&](StringRef name, Attribute attr) {
      auto nameAttr = StringAttr::get(context, name);
      for (auto &item : configItems) {
        // Replace the existing entry.
        if (item.getName() == nameAttr) {
          item.setValue(attr);
          return;
        }
      }
      configItems.push_back(NamedAttribute(nameAttr, attr));
    };
    auto updateConfigItem = [&](IREE::CPU::TilingLevel level, Attribute attr) {
      setConfigItem(IREE::CPU::getTilingLevelName(level), attr);
    };

    auto distParallelAttr = getTilingLevelAttr(context, distParallelSizes);
    updateConfigItem(IREE::CPU::TilingLevel::CacheParallelTiles,
                     distParallelAttr);

    auto cacheReductionAttr = getTilingLevelAttr(context, cacheReductionSizes);
    updateConfigItem(IREE::CPU::TilingLevel::CacheReductionTiles,
                     cacheReductionAttr);

    // Write back the vector tile sizes only if this pass changed them (Zvt
    // sizes or clamped sizes), so that the other attributes of the levels stay.
    if (useZvt || parallelClamped) {
      updateConfigItem(IREE::CPU::TilingLevel::VectorCommonParallelTiles,
                       getTilingLevelAttr(context, vectorParallelSizes));
    }
    if (useZvt || reductionClamped) {
      updateConfigItem(IREE::CPU::TilingLevel::VectorReductionTiles,
                       getTilingLevelAttr(context, vectorReductionSizes));
    }

    // Zvt needs loop peeling, and the other compute ops must follow the Zvt
    // tiles of the root op.
    if (useZvt) {
      // Peel partial DTCM and vector tiles (e.g. 80 by 64): Zvt needs static
      // shapes.
      NamedAttrList config(translationInfo.getConfiguration());
      config.set(getEnableLoopPeelingStr(), UnitAttr::get(context));
      translationInfo = IREE::Codegen::TranslationInfoAttr::get(
          context, translationInfo.getPassPipeline(),
          translationInfo.getCodegenSpec(), translationInfo.getWorkgroupSize(),
          translationInfo.getSubgroupSize(), config.getDictionary(context));
      // setTranslationInfo cannot fail today; propagate a failure anyway.
      if (failed(setTranslationInfo(funcOp, translationInfo))) {
        signalPassFailure();
        return;
      }

      // The strategy pass derived the vector tile sizes of the other compute
      // ops from the RVV tile sizes of the root op. Erase their lowering
      // configs, so that these ops follow the Zvt tiles of the root op through
      // fusion.
      for (Operation *computeOp : getComputeOps(funcOp)) {
        // Keep the lowering config of the root op.
        if (computeOp == tilingOp.getOperation()) continue;
        eraseLoweringConfig(computeOp);
      }
    }

    setLoweringConfig(tilingOp,
                      IREE::CPU::LoweringConfigAttr::get(context, configItems));

    markAnalysesPreserved<CoralNPUTileSizeSelectionAnalysis>();
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
