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
#include <string>
#include <utility>

#include "compiler/Transforms/CoralNPUTileSizeSelectionAnalysis.h"
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
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/Pass/Pass.h"

// LLVM headers
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "coralnpu-tile-size-selection"

using namespace mlir;
using namespace mlir::iree_compiler;

namespace mlir::coralnpu_compiler {

#define GEN_PASS_DEF_CORALNPUTILESIZESELECTIONDTCM
#include "compiler/Transforms/Passes.h.inc"

namespace {

// Estimates the tiled footprint of a single operand of a LinalgOp.
//
// This is not trivial because indexing maps can contain complex affine
// expressions (e.g., `d_spatial + d_filter * dilation` in Conv2D) rather than
// just simple loop dimensions. To estimate the footprint accurately, we must:
// 1. Determine the active range for each loop dimension. If a loop is tiled,
//    its range is its tile size. If it is untiled (tile size is 0), its range
//    is the full static range of the loop.
// 2. For each dimension of the operand's tensor type, we estimate the range of
//    indices accessed by evaluating its indexing AffineExpr.
// 3. We use `mlir::getBoundForAffineExpr` to compute the constant lower and
//    upper bounds of the expression over the loop ranges. The footprint along
//    that dimension is estimated as the span `upper_bound - lower_bound + 1`.
// 4. If the bounds cannot be statically resolved (e.g., due to dynamic shapes
//    or complex symbols), we fallback to the full dimension size of the tensor.
int64_t estimateLinalgOperandFootprint(linalg::LinalgOp linalgOp,
                                       OpOperand *operand,
                                       ArrayRef<int64_t> tileSizes) {
  auto rankedTensorType = dyn_cast<RankedTensorType>(operand->get().getType());
  if (!rankedTensorType) {
    assert(false && "expected a ranked tensor");
    return 0;
  }

  Type elementType = rankedTensorType.getElementType();
  unsigned elementBits;
  if (elementType.isIntOrFloat()) {
    elementBits = elementType.getIntOrFloatBitWidth();
  } else {
    assert(false && "only int/float elements are supported");
    elementBits = 32;
  }

  AffineMap map = linalgOp.getMatchingIndexingMap(operand);
  int64_t numElements = 1;

  SmallVector<int64_t> staticLoopRanges = linalgOp.getStaticLoopRanges();
  assert(tileSizes.size() == staticLoopRanges.size());

  SmallVector<int64_t> resolvedTileSizes(tileSizes);
  for (size_t i = 0; i < resolvedTileSizes.size(); ++i) {
    resolvedTileSizes[i] =
        resolvedTileSizes[i] ? resolvedTileSizes[i] : staticLoopRanges[i];
  }

  SmallVector<std::optional<int64_t>> lowerBounds(resolvedTileSizes.size(), 0);
  SmallVector<std::optional<int64_t>> upperBounds(resolvedTileSizes.size(),
                                                  std::nullopt);
  for (size_t i = 0; i < resolvedTileSizes.size(); ++i) {
    if (resolvedTileSizes[i] > 0) {
      upperBounds[i] = resolvedTileSizes[i] - 1;
    }
  }

  for (auto [i, expr] : llvm::enumerate(map.getResults())) {
    auto lb =
        getBoundForAffineExpr(expr, resolvedTileSizes.size(), 0, lowerBounds,
                              upperBounds, /*isUpper=*/false);
    auto ub = getBoundForAffineExpr(expr, resolvedTileSizes.size(), 0,
                                    lowerBounds, upperBounds, /*isUpper=*/true);
    if (lb && ub) {
      int64_t span = *ub - *lb + 1;
      numElements *= span;
      continue;
    }

    std::string exprStr;
    llvm::raw_string_ostream os(exprStr);
    os << expr;
    linalgOp.emitWarning() << "could not estimate bound for dimension " << i
                           << " of operand #" << operand->getOperandNumber()
                           << " (expression: " << os.str() << ") of '"
                           << linalgOp->getName().getStringRef()
                           << "', falling back to full dimension size ("
                           << rankedTensorType.getDimSize(i)
                           << "). Footprint estimation might be conservative.";
    numElements *= rankedTensorType.getDimSize(i);
  }

  return llvm::divideCeil(numElements * elementBits, 8);
}

// Helper to extract constant inner tile size for a given dimension if present.
int64_t getInnerTileSize(ArrayRef<int64_t> innerDimsPos,
                         ArrayRef<OpFoldResult> mixedTiles, int64_t dim) {
  auto it = llvm::find(innerDimsPos, dim);
  if (it == innerDimsPos.end()) return 1;
  size_t k = std::distance(innerDimsPos.begin(), it);
  return getConstantIntValue(mixedTiles[k]).value_or(1);
}

// Estimates footprint (in elements) of a packed tensor operand (rank N + M).
int64_t estimatePackedOperandFootprint(RankedTensorType type,
                                       ArrayRef<int64_t> tileSizes,
                                       ArrayRef<int64_t> outerDimsPerm,
                                       ArrayRef<int64_t> innerDimsPos,
                                       ArrayRef<OpFoldResult> mixedTiles,
                                       bool isTiledByUnpackedElements) {
  int64_t numElements = 1;
  int64_t rank = type.getRank();
  auto shape = type.getShape();
  int64_t numLoops = tileSizes.size();

  // 1. Outer dims (tiled)
  for (int64_t l = 0; l < numLoops; ++l) {
    int64_t srcDim = outerDimsPerm.empty() ? l : outerDimsPerm[l];

    int64_t tileSize = tileSizes[l];
    if (tileSize == 0) {
      numElements *= shape[srcDim];
      continue;
    }

    if (isTiledByUnpackedElements) {
      int64_t innerTileSize =
          getInnerTileSize(innerDimsPos, mixedTiles, srcDim);
      int64_t neededTiles = llvm::divideCeil(tileSize, innerTileSize);
      numElements *= std::min(shape[srcDim], neededTiles);
      continue;
    }

    numElements *= std::min(shape[srcDim], tileSize);
  }

  // 2. Inner dims (untiled / full)
  for (int64_t d = numLoops; d < rank; ++d) {
    numElements *= shape[d];
  }

  int64_t elementBits = type.getElementType().getIntOrFloatBitWidth();
  return llvm::divideCeil(numElements * elementBits, 8);
}

// Estimates footprint (in elements) of an unpacked tensor operand (rank N).
int64_t estimateUnpackedOperandFootprint(RankedTensorType type,
                                         ArrayRef<int64_t> tileSizes,
                                         ArrayRef<int64_t> outerDimsPerm,
                                         ArrayRef<int64_t> innerDimsPos,
                                         ArrayRef<OpFoldResult> mixedTiles,
                                         bool isTiledByOuterTiles) {
  int64_t numElements = 1;
  auto shape = type.getShape();
  int64_t numLoops = tileSizes.size();

  for (int64_t l = 0; l < numLoops; ++l) {
    int64_t inputDim = outerDimsPerm.empty() ? l : outerDimsPerm[l];
    int64_t tileSize = tileSizes[l];

    if (tileSize == 0) {
      numElements *= shape[inputDim];
      continue;
    }

    if (isTiledByOuterTiles) {
      int64_t innerTileSize =
          getInnerTileSize(innerDimsPos, mixedTiles, inputDim);
      numElements *= std::min(shape[inputDim], tileSize * innerTileSize);
      continue;
    }

    numElements *= std::min(shape[inputDim], tileSize);
  }

  int64_t elementBits = type.getElementType().getIntOrFloatBitWidth();
  return llvm::divideCeil(numElements * elementBits, 8);
}

int64_t estimatePackOperandFootprint(linalg::PackOp packOp, OpOperand *operand,
                                     ArrayRef<int64_t> tileSizes) {
  Value value = operand->get();

  auto rankedTensorType = dyn_cast<RankedTensorType>(value.getType());
  if (!rankedTensorType) {
    assert(false && "expected a ranked tensor");
    return 0;
  }

  if (value == packOp.getSource()) {
    return estimateUnpackedOperandFootprint(
        rankedTensorType, tileSizes, packOp.getOuterDimsPerm(),
        packOp.getInnerDimsPos(), packOp.getMixedTiles(),
        /*isTiledByOuterTiles=*/true);
  }

  if (value == packOp.getDest()) {
    return estimatePackedOperandFootprint(
        rankedTensorType, tileSizes, /*outerDimsPerm=*/{}, /*innerDimsPos=*/{},
        /*mixedTiles=*/{}, /*isTiledByUnpackedElements=*/false);
  }

  Type elementType = rankedTensorType.getElementType();
  unsigned elementBits;
  if (elementType.isIntOrFloat()) {
    elementBits = elementType.getIntOrFloatBitWidth();
  } else {
    assert(false && "only int/float elements are supported");
    elementBits = 32;
  }

  return llvm::divideCeil(rankedTensorType.getNumElements() * elementBits, 8);
}

int64_t estimateUnpackOperandFootprint(linalg::UnPackOp unpackOp,
                                       OpOperand *operand,
                                       ArrayRef<int64_t> tileSizes) {
  Value value = operand->get();

  auto rankedTensorType = dyn_cast<RankedTensorType>(value.getType());
  if (!rankedTensorType) {
    assert(false && "expected a ranked tensor");
    return 0;
  }

  if (value == unpackOp.getDest()) {
    return estimateUnpackedOperandFootprint(
        rankedTensorType, tileSizes, /*outerDimsPerm=*/{}, /*innerDimsPos=*/{},
        /*mixedTiles=*/{}, /*isTiledByOuterTiles=*/false);
  }

  if (value == unpackOp.getSource()) {
    return estimatePackedOperandFootprint(
        rankedTensorType, tileSizes, unpackOp.getOuterDimsPerm(),
        unpackOp.getInnerDimsPos(), unpackOp.getMixedTiles(),
        /*isTiledByUnpackedElements=*/true);
  }

  Type elementType = rankedTensorType.getElementType();
  unsigned elementBits;
  if (elementType.isIntOrFloat()) {
    elementBits = elementType.getIntOrFloatBitWidth();
  } else {
    assert(false && "only int/float elements are supported");
    elementBits = 32;
  }

  return llvm::divideCeil(rankedTensorType.getNumElements() * elementBits, 8);
}

// Estimates the tiled footprint of a single operand of a TilingInterface op.
int64_t estimateOperandFootprint(TilingInterface tilingOp, OpOperand *operand,
                                 ArrayRef<int64_t> tileSizes) {
  Value value = operand->get();

  // Handle scalar operands
  Type type = value.getType();
  if (type.isIntOrFloat()) {
    return llvm::divideCeil(type.getIntOrFloatBitWidth(), 8);
  }

  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(tilingOp.getOperation())) {
    return estimateLinalgOperandFootprint(linalgOp, operand, tileSizes);
  }
  if (auto packOp = dyn_cast<linalg::PackOp>(tilingOp.getOperation())) {
    return estimatePackOperandFootprint(packOp, operand, tileSizes);
  }
  if (auto unpackOp = dyn_cast<linalg::UnPackOp>(tilingOp.getOperation())) {
    return estimateUnpackOperandFootprint(unpackOp, operand, tileSizes);
  }

  // Fallback for non-Linalg operations
  auto rankedTensorType = dyn_cast<RankedTensorType>(value.getType());
  if (!rankedTensorType) {
    assert(false && "expected a ranked tensor");
    return 0;
  }

  int64_t numElements = 1;
  if (rankedTensorType.getRank() == tileSizes.size()) {
    for (auto [dimIdx, dimSize] :
         llvm::enumerate(rankedTensorType.getShape())) {
      int64_t tileSize = tileSizes[dimIdx];
      if (tileSize == 0 || ShapedType::isDynamic(dimSize)) {
        numElements *= ShapedType::isDynamic(dimSize) ? 1 : dimSize;
      } else {
        numElements *= std::min(dimSize, tileSize);
      }
    }
  } else {
    numElements = rankedTensorType.getNumElements();
  }

  Type elementType = rankedTensorType.getElementType();
  unsigned elementBits;
  if (elementType.isIntOrFloat()) {
    elementBits = elementType.getIntOrFloatBitWidth();
  } else {
    assert(false && "only int/float elements are supported");
    elementBits = 32;
  }

  return llvm::divideCeil(numElements * elementBits, 8);
}

// Estimates the footprint of the temporary buffers allocated during unpack
// lowering.
//
// Unpack lowering uses temporary stack buffers in DTCM to perform the transpose
// and unpacking operations. The size and number of these buffers depend on:
// 1. Whether the register-level transpose reuse optimization succeeds (which
//    requires register tile size N to be a multiple of inner tile size N).
//    - If it succeeds, the footprint is small and constant (proportional to
//      register tile sizes).
//    - If it fails, the footprint is larger and proportional to DTCM tile
//    sizes,
//      often requiring 3 separate buffers.
// 2. Vectorization padding (e.g., +2, +10) added to handle unaligned vector
//    accesses.
//
// This function estimates the footprint by resolving the tile sizes to register
// tile sizes (if available) or falling back to DTCM tile sizes, ensuring a safe
// and accurate estimate in both cases.
int64_t estimateUnpackTemporariesFootprint(
    linalg::UnPackOp unpackOp, ArrayRef<int64_t> tileSizes,
    ArrayRef<int64_t> registerTileSizes) {
  auto destType = dyn_cast<RankedTensorType>(unpackOp.getDest().getType());
  if (!destType) return 0;

  int64_t elementBits = destType.getElementType().getIntOrFloatBitWidth();
  int64_t elemSize = llvm::divideCeil(elementBits, 8);

  auto innerDimsPos = unpackOp.getInnerDimsPos();
  auto mixedTiles = unpackOp.getMixedTiles();

  if (innerDimsPos.size() != 2) {
    // Fallback if not 2D unpack
    int64_t totalElements = 1;
    for (size_t i = 0; i < innerDimsPos.size(); ++i) {
      int64_t dim = innerDimsPos[i];
      int64_t tileSize = tileSizes[dim];
      if (dim < registerTileSizes.size() && registerTileSizes[dim] > 0) {
        tileSize = registerTileSizes[dim];
      }
      int64_t innerTileSize = getConstantIntValue(mixedTiles[i]).value_or(1);
      int64_t numTiles = llvm::divideCeil(tileSize, innerTileSize);
      if (tileSize % innerTileSize != 0) numTiles += 1;
      totalElements *= numTiles * innerTileSize;
    }
    return 2 * totalElements * elemSize;
  }

  int64_t dimM = innerDimsPos[0];
  int64_t dimN = innerDimsPos[1];

  int64_t tileM = tileSizes[dimM];
  int64_t tileN = tileSizes[dimN];

  int64_t regM = 0;
  int64_t regN = 0;
  if (dimM < registerTileSizes.size()) {
    regM = registerTileSizes[dimM];
  }
  if (dimN < registerTileSizes.size()) {
    regN = registerTileSizes[dimN];
  }

  // We use register tile sizes (if available) instead of DTCM tile sizes for
  // footprint estimation. In the optimized codegen pipeline, the transpose
  // temporary is bufferized to a register-tile-sized stack allocation that is
  // reused.
  // However, if the register tile size N (regN) is smaller than the inner tile
  // size N (innerN), this reuse optimization fails, and the compiler allocates
  // 3 separate buffers of sizes proportional to the DTCM tile size.
  // Using resolvedTileM/N (which fallback to DTCM tile sizes if register sizes
  // are not yet selected) handles both cases: it correctly estimates the large
  // footprint when reuse fails, and significantly reduces the estimate (making
  // it constant based on register tile size) when reuse succeeds.
  // See unpack_footprint_analysis.md for details.
  int64_t resolvedTileM = regM > 0 ? regM : tileM;
  int64_t resolvedTileN = regN > 0 ? regN : tileN;

  int64_t innerM = getConstantIntValue(mixedTiles[0]).value_or(1);
  int64_t innerN = getConstantIntValue(mixedTiles[1]).value_or(1);

  auto getMaxOuterTiles = [](int64_t tile, int64_t inner) -> int64_t {
    if (tile == 0)
      return int64_t(1);  // If not tiled, we might still need 1 tile?
    if (tile % inner == 0) return tile / inner;
    return llvm::divideCeil(tile, inner) + 1;
  };

  int64_t maxOuterM = getMaxOuterTiles(resolvedTileM, innerM);
  int64_t maxOuterN = getMaxOuterTiles(resolvedTileN, innerN);

  // The magic numbers `+ 2` and `paddingN = 10` (and the `+ 1` in
  // getMaxOuterTiles when not a multiple) represent safety padding added by the
  // compiler for vectorization (to handle boundary conditions and unaligned
  // vector accesses).
  // The 3-buffer structure `buffer1 + 2 * buffer2` represents the worst-case
  // allocations when reuse optimization fails.
  int64_t paddingN = 0;
  if (resolvedTileN > 0 && resolvedTileN % innerN != 0) {
    paddingN = 10;
  }

  int64_t sizeM = maxOuterM * innerM;
  int64_t sizeN = maxOuterN * innerN;

  int64_t buffer1 = innerM * (sizeN + 2) * elemSize;
  int64_t buffer2 = (sizeM + 2) * (sizeN + paddingN) * elemSize;

  return buffer1 + 2 * buffer2;
}

// Estimates the combined footprint of the op's operands.
int64_t estimateFootprint(TilingInterface tilingOp,
                          ArrayRef<int64_t> tileSizes) {
  int64_t totalBytes = 0;

  for (OpOperand &operand : tilingOp->getOpOperands()) {
    totalBytes += estimateOperandFootprint(tilingOp, &operand, tileSizes);
  }

  if (auto unpackOp = dyn_cast<linalg::UnPackOp>(tilingOp.getOperation())) {
    SmallVector<int64_t> registerTileSizes;
    IREE::Codegen::LoweringConfigAttrInterface loweringConfig;
    if (auto compilationInfo = getCompilationInfo(tilingOp)) {
      loweringConfig = compilationInfo.getLoweringConfig();
    } else {
      loweringConfig = getLoweringConfig(tilingOp);
    }
    if (loweringConfig) {
      auto regLevel = IREE::CPU::TilingLevel::VectorCommonParallelTiles;
      registerTileSizes = loweringConfig.getStaticTilingLevelSizes(
          static_cast<unsigned>(regLevel), tilingOp);
    }
    totalBytes += estimateUnpackTemporariesFootprint(unpackOp, tileSizes,
                                                     registerTileSizes);
  }

  return totalBytes;
}

bool shrinkLoops(ArrayRef<size_t> loopIndices, ArrayRef<int64_t> alignments,
                 MutableArrayRef<int64_t> dtcmTileSizes) {
  for (size_t loopIdx : llvm::reverse(loopIndices)) {
    if (alignments[loopIdx] <= 0) continue;
    if (dtcmTileSizes[loopIdx] < 2 * alignments[loopIdx]) continue;
    dtcmTileSizes[loopIdx] =
        llvm::alignDown(dtcmTileSizes[loopIdx] / 2, alignments[loopIdx]);
    return true;
  }
  return false;
}

void alignTileSizes(ArrayRef<size_t> loopIndices, ArrayRef<int64_t> alignments,
                    MutableArrayRef<int64_t> dtcmTileSizes) {
  for (size_t loopIdx : loopIndices) {
    if (alignments[loopIdx] <= 0) continue;
    if (dtcmTileSizes[loopIdx] <= alignments[loopIdx]) continue;
    dtcmTileSizes[loopIdx] =
        llvm::alignDown(dtcmTileSizes[loopIdx], alignments[loopIdx]);
  }
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

    LLVM_DEBUG(llvm::dbgs()
               << "Running CoralNPUTileSizeSelectionDTCMPass (DTCM Size: "
               << dtcmSizeKb << " KB) on function: " << funcOp.getName()
               << "\n");

    if (dtcmSizeKb <= 0) {
      funcOp.emitError("dtcm-size-kb must be positive, got ") << dtcmSizeKb;
      signalPassFailure();
      return;
    }

    auto &analysis = getAnalysis<CoralNPUTileSizeSelectionAnalysis>();
    if (failed(analysis.status)) {
      return;
    }

    auto tilingOp = analysis.rootTilingOp;

    auto compilationInfo = getCompilationInfo(tilingOp);
    if (!compilationInfo) {
      return;
    }

    auto cpuLoweringConfig = dyn_cast<IREE::CPU::LoweringConfigAttr>(
        compilationInfo.getLoweringConfig());
    if (!cpuLoweringConfig) {
      tilingOp->emitOpError("expected CPU lowering config");
      signalPassFailure();
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

    SmallVector<int64_t> dtcmTileSizes(analysis.staticLoopRanges);

    alignTileSizes(analysis.parallelLoops, vectorParallelSizes, dtcmTileSizes);

    alignTileSizes(analysis.reductionLoops, vectorReductionSizes,
                   dtcmTileSizes);

    // TODO(sflur): tune the safetyMultiplier (do we want a commandline option
    // for it?).
    const double safetyMultiplier = 1.2;
    bool fallback = false;
    while (estimateFootprint(tilingOp, dtcmTileSizes) * safetyMultiplier >
           dtcmSizeKb * 1024) {
      if (shrinkLoops(analysis.parallelLoops, vectorParallelSizes,
                      dtcmTileSizes))
        continue;

      if (shrinkLoops(analysis.reductionLoops, vectorReductionSizes,
                      dtcmTileSizes))
        continue;

      // If any of the align sizes are 0, their dimensions were not shrunken;
      // set these alignments to 1, and try to shrink again.
      bool unalignedFallback = false;
      for (auto &size : vectorParallelSizes) {
        if (size <= 0) {
          size = 1;
          unalignedFallback = true;
        }
      }
      for (auto &size : vectorReductionSizes) {
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

        vectorParallelSizes.assign(vectorParallelSizes.size(), 1);
        vectorReductionSizes.assign(vectorReductionSizes.size(), 1);

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

    LLVM_DEBUG({
      llvm::dbgs() << "    Selected DTCM tile sizes: [";
      for (auto size : dtcmTileSizes) llvm::dbgs() << size << " ";
      llvm::dbgs() << "]\n";
      llvm::dbgs() << "    Estimated footprint: "
                   << estimateFootprint(tilingOp, dtcmTileSizes) << " bytes\n";
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

    auto updateConfigItem = [&](IREE::CPU::TilingLevel level, Attribute attr) {
      auto name =
          StringAttr::get(context, IREE::CPU::getTilingLevelName(level));
      for (auto &item : configItems) {
        if (item.getName() == name) {
          item.setValue(attr);
          return;
        }
      }
      configItems.push_back(NamedAttribute(name, attr));
    };

    auto distParallelAttr = getTilingLevelAttr(context, distParallelSizes);
    updateConfigItem(IREE::CPU::TilingLevel::CacheParallelTiles,
                     distParallelAttr);

    auto cacheReductionAttr = getTilingLevelAttr(context, cacheReductionSizes);
    updateConfigItem(IREE::CPU::TilingLevel::CacheReductionTiles,
                     cacheReductionAttr);

    auto newLoweringConfig =
        IREE::CPU::LoweringConfigAttr::get(context, configItems);

    auto newCompilationInfo = IREE::Codegen::CompilationInfoAttr::get(
        context, newLoweringConfig, compilationInfo.getTranslationInfo());

    setCompilationInfo(tilingOp, newCompilationInfo);

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
