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
// CoralNPU Tile Size Selection Pass
//
// This pass determines the optimal tile sizes for CoralNPU across all tiling
// levels (Workgroups, L2 Cache, L1/DTCM, and Registers).
//
// For a detailed explanation of the tiling levels and a step-by-step guided
// example of how a large matrix multiplication is tiled, see:
// docs/design/tiling_levels_explained.md
//===----------------------------------------------------------------------===//

#include "compiler/Transforms/Passes.h"

// IREE headers
#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUDialect.h"
#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUTypes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Utils/CPUUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"

// MLIR headers
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/TypeID.h"

// LLVM headers
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "coralnpu-tile-size-selection"

using namespace mlir;
using namespace mlir::iree_compiler;

namespace mlir::coralnpu_compiler {

#define GEN_PASS_DEF_CORALNPUTILESIZESELECTIONREGISTER
#define GEN_PASS_DEF_CORALNPUTILESIZESELECTIONDTCM
#define GEN_PASS_DEF_CORALNPUTILESIZESELECTIONWORKGROUP
#include "compiler/Transforms/Passes.h.inc"

namespace {

// Helper to create a tiling level attribute with default (false) scalable
// flags. This is needed to prevent crashes in IREE passes (e.g.
// LLVMCPUSplitReduction) that assume scalable flags are present.
Attribute getTilingLevelAttr(MLIRContext *context, ArrayRef<int64_t> sizes) {
  SmallVector<bool> scalableFlags(sizes.size(), false);
  return IREE::CPU::LoweringConfigAttr::getTilingLevelAttr(context, sizes,
                                                           scalableFlags);
}

// Classifies loops of a TilingInterface operation into logical categories
// to apply target-specific tiling alignments.
//
// Example for Matmul (M x N x K):
// - M-Loop (LHS rows == Output rows) -> vectorLoops (Parallel)
// - N-Loop (RHS cols == Output cols) -> unrollLoops (Parallel)
// - K-Loop (LHS cols == RHS rows)    -> reductionLoops (Reduction)
//
// Example for 2D Convolution (N x OH x OW x OC x KH x KW x IC):
// - Batch (N) and Spatial (OH, OW) loops -> vectorLoops (parallel)
// - Output Channel (OC) loop             -> unrollLoops (parallel)
// - Filter (KH, KW) and Input Channel (IC) loops -> reductionLoops (reduction)
struct LoopClassification {
  // Parallel loops associated primarily with the LHS or batch/spatial
  // dimensions. Typically aligned to the hardware vector length (VLEN).
  SmallVector<size_t> vectorLoops;

  // Parallel loops associated primarily with the RHS or output channel
  // dimensions. Typically aligned to a multiple of VLEN (vector width * unroll
  // factor) to maximize register reuse.
  SmallVector<size_t> unrollLoops;

  // Reduction loops. Typically tiled to 1 at the register level.
  SmallVector<size_t> reductionLoops;
};

// Refines parallel loops for Linalg operations to distinguish M and N loops
// for matmul-like alignment.
void refineLinalgOpLoops(linalg::LinalgOp linalgOp,
                         LoopClassification &classification) {
  if (linalgOp.getNumDpsInputs() < 2) return;

  auto lhsMap = linalgOp.getMatchingIndexingMap(linalgOp.getDpsInputOperand(0));
  auto rhsMap = linalgOp.getMatchingIndexingMap(linalgOp.getDpsInputOperand(1));

  SmallVector<size_t> parallelLoops;
  std::swap(parallelLoops, classification.vectorLoops);

  for (size_t parallelLoop : parallelLoops) {
    if (lhsMap.isFunctionOfDim(parallelLoop)) {
      classification.vectorLoops.push_back(parallelLoop);
      continue;
    }

    if (rhsMap.isFunctionOfDim(parallelLoop)) {
      classification.unrollLoops.push_back(parallelLoop);
      continue;
    }

    classification.vectorLoops.push_back(parallelLoop);
  }
}

// Classifies loops of a TilingInterface op.
LoopClassification classifyLoops(TilingInterface tilingOp) {
  LoopClassification classification;
  auto iterTypes = tilingOp.getLoopIteratorTypes();

  for (auto [index, iterType] : llvm::enumerate(iterTypes)) {
    if (iterType == utils::IteratorType::reduction) {
      classification.reductionLoops.push_back(index);
      continue;
    }

    assert(iterType == utils::IteratorType::parallel);
    classification.vectorLoops.push_back(index);
  }

  // If it is a LinalgOp, we can try to distinguish M and N for matmul-like
  // alignment
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(tilingOp.getOperation())) {
    refineLinalgOpLoops(linalgOp, classification);
    return classification;
  }

  return classification;
}

// Helper to get static loop ranges from TilingInterface
SmallVector<int64_t> getStaticLoopRanges(TilingInterface op) {
  OpBuilder builder(op);
  SmallVector<Range> loopRange = op.getIterationDomain(builder);
  return llvm::map_to_vector(loopRange, [](Range r) -> int64_t {
    std::optional<int64_t> intVal = getConstantIntValue(r.size);
    return intVal ? intVal.value() : ShapedType::kDynamic;
  });
}

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
    return 0;
  }

  Type elementType = rankedTensorType.getElementType();
  unsigned elementBits = elementType.getIntOrFloatBitWidth();

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
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type) return 0;

  if (value == packOp.getSource()) {
    return estimateUnpackedOperandFootprint(
        type, tileSizes, packOp.getOuterDimsPerm(), packOp.getInnerDimsPos(),
        packOp.getMixedTiles(), /*isTiledByOuterTiles=*/true);
  }

  if (value == packOp.getDest()) {
    return estimatePackedOperandFootprint(
        type, tileSizes, /*outerDimsPerm=*/{}, /*innerDimsPos=*/{},
        /*mixedTiles=*/{}, /*isTiledByUnpackedElements=*/false);
  }

  int64_t elementBits = type.getElementType().getIntOrFloatBitWidth();
  return llvm::divideCeil(type.getNumElements() * elementBits, 8);
}

int64_t estimateUnpackOperandFootprint(linalg::UnPackOp unpackOp,
                                       OpOperand *operand,
                                       ArrayRef<int64_t> tileSizes) {
  Value value = operand->get();
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type) return 0;

  if (value == unpackOp.getDest()) {
    return estimateUnpackedOperandFootprint(
        type, tileSizes, /*outerDimsPerm=*/{}, /*innerDimsPos=*/{},
        /*mixedTiles=*/{}, /*isTiledByOuterTiles=*/false);
  }

  if (value == unpackOp.getSource()) {
    return estimatePackedOperandFootprint(
        type, tileSizes, unpackOp.getOuterDimsPerm(),
        unpackOp.getInnerDimsPos(), unpackOp.getMixedTiles(),
        /*isTiledByUnpackedElements=*/true);
  }

  int64_t elementBits = type.getElementType().getIntOrFloatBitWidth();
  return llvm::divideCeil(type.getNumElements() * elementBits, 8);
}

// Estimates the tiled footprint of a single operand of a TilingInterface op.
int64_t estimateOperandFootprint(TilingInterface tilingOp, OpOperand *operand,
                                 ArrayRef<int64_t> tileSizes) {
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
  Value value = operand->get();
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type) return 0;

  int64_t numElements = 1;
  if (type.getRank() == static_cast<int64_t>(tileSizes.size())) {
    for (auto [dimIdx, dimSize] : llvm::enumerate(type.getShape())) {
      int64_t tileSize = tileSizes[dimIdx];
      if (tileSize == 0 || ShapedType::isDynamic(dimSize)) {
        numElements *= ShapedType::isDynamic(dimSize) ? 1 : dimSize;
      } else {
        numElements *= std::min(dimSize, tileSize);
      }
    }
  } else {
    for (int64_t dimSize : type.getShape()) {
      numElements *= ShapedType::isDynamic(dimSize) ? 1 : dimSize;
    }
  }
  return numElements * (type.getElementType().getIntOrFloatBitWidth() / 8);
}

// Estimates the combined footprint of the op's operands.
int64_t estimateFootprint(TilingInterface tilingOp,
                          ArrayRef<int64_t> tileSizes) {
  int64_t totalBytes = 0;
  for (OpOperand &operand : tilingOp.getOperation()->getOpOperands()) {
    totalBytes += estimateOperandFootprint(tilingOp, &operand, tileSizes);
  }
  return totalBytes;
}

// Parses VLEN from target CPU features (+zvl<N>b).
FailureOr<int64_t> getVlenFromTargetFeatures(FunctionOpInterface funcOp) {
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(funcOp);
  if (!targetAttr) return failure();

  auto config = targetAttr.getConfiguration();
  if (!config) return failure();

  auto attr = config.getAs<StringAttr>("cpu_features");
  if (!attr) return failure();

  llvm::StringRef cpuFeatures = attr.getValue();
  size_t pos = cpuFeatures.find("+zvl");
  if (pos == llvm::StringRef::npos) return failure();

  llvm::StringRef suffix = cpuFeatures.substr(pos + 4);
  size_t endPos = suffix.find("b");
  if (endPos == llvm::StringRef::npos) return failure();

  llvm::StringRef vlenStr = suffix.substr(0, endPos);
  int64_t parsedVlen = 0;
  if (vlenStr.getAsInteger(10, parsedVlen)) return failure();

  return parsedVlen;
}

struct Alignments {
  int64_t vectorAlign;
  int64_t unrollAlign;
  int64_t reductionAlign;
};

// Computes optimal alignments dynamically based on register budget and vector
// width.
void resolveAlignments(int64_t vectorWidth, int64_t numRegisters,
                       const LoopClassification &loops,
                       ArrayRef<int64_t> staticLoopRanges,
                       int64_t parallelAlignmentOpt, Alignments &alignments) {
  if (loops.unrollLoops.empty()) {
    alignments.unrollAlign = 0;

    if (parallelAlignmentOpt != 0) {
      alignments.vectorAlign = parallelAlignmentOpt;
      return;
    }

    if (alignments.vectorAlign != 0) return;

    alignments.vectorAlign = vectorWidth;
    return;
  }

  // Cap M_reg at min(vectorWidth, 8) to avoid exceeding register file for large
  // vector widths (like i8).
  if (alignments.vectorAlign == 0) {
    // TODO (sflur): the 8 below is roughly `numRegisters / 4`, so maybe we
    // should do that instead of 8.
    int64_t defaultMReg = std::min<int64_t>(vectorWidth, 8);
    alignments.vectorAlign = defaultMReg;
  }

  if (alignments.unrollAlign != 0) return;

  // Compute unroll factor U based on register budget:
  // 1 (LHS) + U (RHS) + M_reg * U (Acc) <= numRegisters
  // U * (1 + M_reg) <= numRegisters - 1
  // U <= (numRegisters - 1) / (1 + M_reg)
  int64_t uLimit = (numRegisters - 1) / (1 + alignments.vectorAlign);
  uLimit = std::max<int64_t>(1, uLimit);

  // Consider actual unroll loop sizes to avoid over-aligning.
  // If there are multiple unroll loops, we take the maximum required unroll
  // factor so that larger loops can utilize the full budget, while smaller
  // loops will be capped by their static range during tiling.
  int64_t uOpt = uLimit;
  if (!loops.unrollLoops.empty()) {
    int64_t maxFullUnroll = 0;
    for (size_t unrollLoopIdx : loops.unrollLoops) {
      int64_t unrollLoopSize = staticLoopRanges[unrollLoopIdx];
      assert(unrollLoopSize > 0);
      int64_t fullUnroll = llvm::divideCeil(unrollLoopSize, vectorWidth);
      maxFullUnroll = std::max(maxFullUnroll, fullUnroll);
    }
    assert(maxFullUnroll > 0);
    uOpt = std::min(uLimit, maxFullUnroll);
  }

  alignments.unrollAlign = vectorWidth * uOpt;
  return;
}

bool shrinkLoops(ArrayRef<size_t> loopIndices, int64_t alignment,
                 SmallVectorImpl<int64_t> &dtcmTileSizes) {
  for (size_t loopIdx : loopIndices) {
    if (dtcmTileSizes[loopIdx] < 2 * alignment) continue;
    dtcmTileSizes[loopIdx] =
        llvm::alignDown(dtcmTileSizes[loopIdx] / 2, alignment);
    return true;
  }
  return false;
}

void alignTileSizes(ArrayRef<size_t> loopIndices, int64_t alignment,
                    SmallVectorImpl<int64_t> &dtcmTileSizes) {
  for (size_t loopIdx : loopIndices) {
    if (dtcmTileSizes[loopIdx] <= alignment) continue;
    dtcmTileSizes[loopIdx] = llvm::alignDown(dtcmTileSizes[loopIdx], alignment);
  }
}

struct CoralNPUTilingAnalysis {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CoralNPUTilingAnalysis)

  TilingInterface rootTilingOp = nullptr;
  // TODO: handle sub-byte elements
  int64_t elemSizeBytes = 0;
  SmallVector<int64_t> staticLoopRanges;
  LoopClassification classification;
  LogicalResult status = failure();

  explicit CoralNPUTilingAnalysis(Operation *op) {
    auto funcOp = dyn_cast<FunctionOpInterface>(op);
    if (!funcOp) return;

    SmallVector<Operation *> computeOps = getComputeOps(funcOp);
    auto rootOpOr = getRootOperation(computeOps);
    if (failed(rootOpOr) || !rootOpOr.value()) return;

    rootTilingOp = dyn_cast<TilingInterface>(*rootOpOr);
    if (!rootTilingOp) {
      rootOpOr.value()->emitWarning(
          "root operation is not a TilingInterface; skipping tile size "
          "selection");
      return;
    }

    // Element size
    if (rootTilingOp->getNumResults() > 0) {
      if (auto type =
              dyn_cast<ShapedType>(rootTilingOp->getResult(0).getType())) {
        auto elemType = type.getElementType();
        if (!elemType.isIntOrFloat()) {
          rootTilingOp->emitWarning(
              "only integer and float types are supported");
          return;
        }
        auto elemBitWidth = elemType.getIntOrFloatBitWidth();
        if (elemBitWidth < 8) {
          rootTilingOp->emitWarning(
              "sub-byte types (e.g. i1, i4) are not supported");
          return;
        }
        elemSizeBytes = elemBitWidth / 8;
      }
    }

    // Static loop ranges
    staticLoopRanges = getStaticLoopRanges(rootTilingOp);
    if (llvm::is_contained(staticLoopRanges, ShapedType::kDynamic)) {
      rootTilingOp->emitWarning("dynamic shapes are not supported");
      return;
    }

    // Loop classification
    classification = classifyLoops(rootTilingOp);

    status = success();
  }
};

struct CoralNPUTileSizeSelectionRegisterPass
    : public impl::CoralNPUTileSizeSelectionRegisterBase<
          CoralNPUTileSizeSelectionRegisterPass> {
  using CoralNPUTileSizeSelectionRegisterBase::
      CoralNPUTileSizeSelectionRegisterBase;

  void runOnOperation() override {
    auto funcOp = getOperation();
    MLIRContext *context = &getContext();

    auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(funcOp);
    if (!targetAttr || targetAttr.getBackend().getValue() != "coralnpu") {
      return;
    }

    LLVM_DEBUG(llvm::dbgs()
               << "Running CoralNPUTileSizeSelectionRegisterPass on function: "
               << funcOp.getName() << "\n");

    if (numVectorRegisters <= 0) {
      funcOp.emitError("num-vector-registers must be positive, got ")
          << numVectorRegisters;
      signalPassFailure();
      return;
    }
    if (vectorRegisterWidthBits < 0) {
      funcOp.emitError("vector-register-width-bits must be non-negative, got ")
          << vectorRegisterWidthBits;
      signalPassFailure();
      return;
    }
    if (vectorAlignment < 0) {
      funcOp.emitError("vector-alignment must be non-negative, got ")
          << vectorAlignment;
      signalPassFailure();
      return;
    }
    if (unrollAlignment < 0) {
      funcOp.emitError("unroll-alignment must be non-negative, got ")
          << unrollAlignment;
      signalPassFailure();
      return;
    }
    if (reductionAlignment <= 0) {
      funcOp.emitError("reduction-alignment must be positive, got ")
          << reductionAlignment;
      signalPassFailure();
      return;
    }
    if (parallelAlignment < 0) {
      funcOp.emitError("parallel-alignment must be non-negative, got ")
          << parallelAlignment;
      signalPassFailure();
      return;
    }

    auto &analysis = getAnalysis<CoralNPUTilingAnalysis>();
    if (failed(analysis.status)) {
      signalPassFailure();
      return;
    }

    auto tilingOp = analysis.rootTilingOp;
    int64_t elemSizeBytes = analysis.elemSizeBytes;

    // 2. Resolve VLEN and compute vector width
    int64_t vlenBits = vectorRegisterWidthBits;
    if (vlenBits == 0) {
      auto parsedVlen = getVlenFromTargetFeatures(funcOp);
      vlenBits = parsedVlen.value_or(128);
    }
    int64_t vectorWidth = (vlenBits / 8) / elemSizeBytes;
    assert(vectorWidth > 0);

    // 3. Resolve alignments
    Alignments alignments = {vectorAlignment, unrollAlignment,
                             reductionAlignment};
    resolveAlignments(vectorWidth, numVectorRegisters, analysis.classification,
                      analysis.staticLoopRanges, parallelAlignment, alignments);

    // 4. Compute register tile sizes
    int64_t numLoops = analysis.staticLoopRanges.size();
    SmallVector<int64_t> vectorParallelSizes(numLoops, 0);
    SmallVector<int64_t> vectorReductionSizes(numLoops, 0);

    for (size_t vectorLoopIdx : analysis.classification.vectorLoops) {
      vectorParallelSizes[vectorLoopIdx] = std::min<int64_t>(
          analysis.staticLoopRanges[vectorLoopIdx], alignments.vectorAlign);
    }

    for (size_t unrollLoopIdx : analysis.classification.unrollLoops) {
      vectorParallelSizes[unrollLoopIdx] = std::min<int64_t>(
          analysis.staticLoopRanges[unrollLoopIdx], alignments.unrollAlign);
    }

    for (size_t reductionLoopIdx : analysis.classification.reductionLoops) {
      vectorReductionSizes[reductionLoopIdx] =
          std::min<int64_t>(analysis.staticLoopRanges[reductionLoopIdx],
                            alignments.reductionAlign);
    }

    // 5. Set in lowering config
    auto vectorParallelAttr = getTilingLevelAttr(context, vectorParallelSizes);
    auto vectorReductionAttr =
        getTilingLevelAttr(context, vectorReductionSizes);

    SmallVector<NamedAttribute> configItems;
    configItems.push_back(NamedAttribute(
        StringAttr::get(context,
                        IREE::CPU::getTilingLevelName(
                            IREE::CPU::TilingLevel::VectorCommonParallelTiles)),
        vectorParallelAttr));
    configItems.push_back(NamedAttribute(
        StringAttr::get(context,
                        IREE::CPU::getTilingLevelName(
                            IREE::CPU::TilingLevel::VectorReductionTiles)),
        vectorReductionAttr));

    auto loweringConfig =
        IREE::CPU::LoweringConfigAttr::get(context, configItems);

    auto translationInfo = IREE::Codegen::TranslationInfoAttr::get(
        context,
        IREE::Codegen::DispatchLoweringPassPipeline::CPUDoubleTilingExpert);

    auto compilationInfo = IREE::Codegen::CompilationInfoAttr::get(
        context, loweringConfig, translationInfo);

    setCompilationInfo(tilingOp.getOperation(), compilationInfo);
    markAnalysesPreserved<CoralNPUTilingAnalysis>();
  }
};

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

    int64_t dtcmLimitBytes = dtcmSizeKb * 1024;

    auto &analysis = getAnalysis<CoralNPUTilingAnalysis>();
    if (failed(analysis.status)) {
      signalPassFailure();
      return;
    }

    auto tilingOp = analysis.rootTilingOp;

    auto compilationInfo = getCompilationInfo(tilingOp.getOperation());
    if (!compilationInfo) {
      tilingOp->emitOpError(
          "missing compilation info from register tiling pass");
      signalPassFailure();
      return;
    }

    auto loweringConfig = compilationInfo.getLoweringConfig();
    auto cpuLoweringConfig =
        dyn_cast<IREE::CPU::LoweringConfigAttr>(loweringConfig);
    if (!cpuLoweringConfig) {
      tilingOp->emitOpError("expected CPU lowering config");
      signalPassFailure();
      return;
    }

    Alignments alignments = {1, 1, 1};

    auto vectorParallelLevel =
        IREE::CPU::TilingLevel::VectorCommonParallelTiles;
    if (cpuLoweringConfig.hasTilingLevel(
            static_cast<unsigned>(vectorParallelLevel))) {
      auto sizes = cpuLoweringConfig.getStaticTilingLevelSizes(
          static_cast<unsigned>(vectorParallelLevel), tilingOp.getOperation());
      for (size_t idx : analysis.classification.vectorLoops) {
        if (idx < sizes.size() && sizes[idx] > 0) {
          alignments.vectorAlign = std::max(alignments.vectorAlign, sizes[idx]);
        }
      }
      for (size_t idx : analysis.classification.unrollLoops) {
        if (idx < sizes.size() && sizes[idx] > 0) {
          alignments.unrollAlign = std::max(alignments.unrollAlign, sizes[idx]);
        }
      }
    }

    auto vectorReductionLevel = IREE::CPU::TilingLevel::VectorReductionTiles;
    if (cpuLoweringConfig.hasTilingLevel(
            static_cast<unsigned>(vectorReductionLevel))) {
      auto sizes = cpuLoweringConfig.getStaticTilingLevelSizes(
          static_cast<unsigned>(vectorReductionLevel), tilingOp.getOperation());
      for (size_t idx : analysis.classification.reductionLoops) {
        if (idx < sizes.size() && sizes[idx] > 0) {
          alignments.reductionAlign =
              std::max(alignments.reductionAlign, sizes[idx]);
        }
      }
    }

    SmallVector<int64_t> dtcmTileSizes(analysis.staticLoopRanges);

    // TODO: check (i.e. benchmark) that this is actually helpful, especially
    // when the operation fits in DTCM without tiling.
    alignTileSizes(analysis.classification.vectorLoops, alignments.vectorAlign,
                   dtcmTileSizes);
    alignTileSizes(analysis.classification.unrollLoops, alignments.unrollAlign,
                   dtcmTileSizes);
    alignTileSizes(analysis.classification.reductionLoops,
                   alignments.reductionAlign, dtcmTileSizes);

    // TODO: tune the safetyMultiplier (do we want a commandline option for
    // it?).
    double safetyMultiplier = 1.2;
    bool fallback = false;
    while (static_cast<double>(estimateFootprint(tilingOp, dtcmTileSizes)) *
               safetyMultiplier >
           dtcmLimitBytes) {
      if (shrinkLoops(analysis.classification.vectorLoops,
                      alignments.vectorAlign, dtcmTileSizes))
        continue;

      if (shrinkLoops(analysis.classification.unrollLoops,
                      alignments.unrollAlign, dtcmTileSizes))
        continue;

      if (shrinkLoops(analysis.classification.reductionLoops,
                      alignments.reductionAlign, dtcmTileSizes))
        continue;

      if (alignments.vectorAlign > 1 || alignments.unrollAlign > 1 ||
          alignments.reductionAlign > 1) {
        tilingOp->emitWarning(llvm::formatv(
            "workload cannot fit in DTCM with resolved alignments (vector={0}, "
            "unroll={1}, reduction={2}); falling back to alignment 1",
            alignments.vectorAlign, alignments.unrollAlign,
            alignments.reductionAlign));

        alignments.vectorAlign = 1;
        alignments.unrollAlign = 1;
        alignments.reductionAlign = 1;
        fallback = true;
        dtcmTileSizes = analysis.staticLoopRanges;
        continue;
      }

      tilingOp->emitOpError(
          "workload cannot fit in DTCM even with 1x1x... tile size");
      signalPassFailure();
      return;
    }

    LLVM_DEBUG({
      llvm::dbgs() << "    Selected DTCM tile sizes: [";
      for (auto size : dtcmTileSizes) llvm::dbgs() << size << " ";
      llvm::dbgs() << "]\n";
      llvm::dbgs() << "    Estimated footprint: "
                   << estimateFootprint(tilingOp, dtcmTileSizes) << " bytes\n";
    });

    int64_t numLoops = analysis.staticLoopRanges.size();
    SmallVector<int64_t> distParallelSizes(numLoops, 0);
    SmallVector<int64_t> cacheReductionSizes(numLoops, 0);

    for (size_t vectorLoopIdx : analysis.classification.vectorLoops) {
      distParallelSizes[vectorLoopIdx] = dtcmTileSizes[vectorLoopIdx];
    }

    for (size_t unrollLoopIdx : analysis.classification.unrollLoops) {
      distParallelSizes[unrollLoopIdx] = dtcmTileSizes[unrollLoopIdx];
    }

    for (size_t reductionLoopIdx : analysis.classification.reductionLoops) {
      cacheReductionSizes[reductionLoopIdx] = dtcmTileSizes[reductionLoopIdx];
    }

    auto distParallelAttr = getTilingLevelAttr(context, distParallelSizes);
    auto cacheReductionAttr = getTilingLevelAttr(context, cacheReductionSizes);

    SmallVector<NamedAttribute> configItems;
    DictionaryAttr oldDict = cpuLoweringConfig.getConfig();
    for (auto attr : oldDict.getValue()) {
      configItems.push_back(attr);
    }

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

    updateConfigItem(IREE::CPU::TilingLevel::CacheParallelTiles,
                     distParallelAttr);
    updateConfigItem(IREE::CPU::TilingLevel::CacheReductionTiles,
                     cacheReductionAttr);

    if (fallback) {
      SmallVector<int64_t> vectorParallelSizes(numLoops, 0);
      SmallVector<int64_t> vectorReductionSizes(numLoops, 0);
      for (size_t vectorLoopIdx : analysis.classification.vectorLoops) {
        vectorParallelSizes[vectorLoopIdx] =
            std::min<int64_t>(dtcmTileSizes[vectorLoopIdx], 1);
      }
      for (size_t unrollLoopIdx : analysis.classification.unrollLoops) {
        vectorParallelSizes[unrollLoopIdx] =
            std::min<int64_t>(dtcmTileSizes[unrollLoopIdx], 1);
      }
      for (size_t reductionLoopIdx : analysis.classification.reductionLoops) {
        vectorReductionSizes[reductionLoopIdx] =
            std::min<int64_t>(dtcmTileSizes[reductionLoopIdx], 1);
      }
      auto vectorParallelAttr =
          getTilingLevelAttr(context, vectorParallelSizes);
      auto vectorReductionAttr =
          getTilingLevelAttr(context, vectorReductionSizes);
      updateConfigItem(IREE::CPU::TilingLevel::VectorCommonParallelTiles,
                       vectorParallelAttr);
      updateConfigItem(IREE::CPU::TilingLevel::VectorReductionTiles,
                       vectorReductionAttr);
    }

    auto newLoweringConfig =
        IREE::CPU::LoweringConfigAttr::get(context, configItems);

    auto newCompilationInfo = IREE::Codegen::CompilationInfoAttr::get(
        context, newLoweringConfig, compilationInfo.getTranslationInfo());

    setCompilationInfo(tilingOp.getOperation(), newCompilationInfo);
    markAnalysesPreserved<CoralNPUTilingAnalysis>();
  }
};

struct CoralNPUTileSizeSelectionWorkgroupPass
    : public impl::CoralNPUTileSizeSelectionWorkgroupBase<
          CoralNPUTileSizeSelectionWorkgroupPass> {
  using CoralNPUTileSizeSelectionWorkgroupBase::
      CoralNPUTileSizeSelectionWorkgroupBase;

  void runOnOperation() override {
    auto funcOp = getOperation();
    MLIRContext *context = &getContext();

    auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(funcOp);
    if (!targetAttr || targetAttr.getBackend().getValue() != "coralnpu") {
      return;
    }

    auto &analysis = getAnalysis<CoralNPUTilingAnalysis>();
    if (failed(analysis.status)) {
      signalPassFailure();
      return;
    }

    auto tilingOp = analysis.rootTilingOp;

    auto compilationInfo = getCompilationInfo(tilingOp.getOperation());
    if (!compilationInfo) {
      tilingOp->emitOpError("missing compilation info");
      signalPassFailure();
      return;
    }

    auto loweringConfig = compilationInfo.getLoweringConfig();
    auto cpuLoweringConfig =
        dyn_cast<IREE::CPU::LoweringConfigAttr>(loweringConfig);
    if (!cpuLoweringConfig) {
      tilingOp->emitOpError("expected CPU lowering config");
      signalPassFailure();
      return;
    }

    auto cacheParallelLevel = IREE::CPU::TilingLevel::CacheParallelTiles;
    if (!cpuLoweringConfig.hasTilingLevel(
            static_cast<unsigned>(cacheParallelLevel))) {
      tilingOp->emitOpError("missing CacheParallelTiles config");
      signalPassFailure();
      return;
    }

    auto cacheParallelAttr = cpuLoweringConfig.getTilingLevelAttr(
        static_cast<unsigned>(cacheParallelLevel));

    SmallVector<NamedAttribute> configItems;
    DictionaryAttr oldDict = cpuLoweringConfig.getConfig();
    for (auto attr : oldDict.getValue()) {
      configItems.push_back(attr);
    }

    auto name = StringAttr::get(context,
                                IREE::CPU::getTilingLevelName(
                                    IREE::CPU::TilingLevel::DistributionTiles));
    bool found = false;
    for (auto &item : configItems) {
      if (item.getName() == name) {
        item.setValue(cacheParallelAttr);
        found = true;
        break;
      }
    }
    if (!found) {
      configItems.push_back(NamedAttribute(name, cacheParallelAttr));
    }

    auto newLoweringConfig =
        IREE::CPU::LoweringConfigAttr::get(context, configItems);

    auto newCompilationInfo = IREE::Codegen::CompilationInfoAttr::get(
        context, newLoweringConfig, compilationInfo.getTranslationInfo());

    setCompilationInfo(tilingOp.getOperation(), newCompilationInfo);
    markAnalysesPreserved<CoralNPUTilingAnalysis>();
  }
};

}  // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createCoralNPUTileSizeSelectionRegisterPass() {
  return std::make_unique<CoralNPUTileSizeSelectionRegisterPass>();
}

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createCoralNPUTileSizeSelectionRegisterPass(
    CoralNPUTileSizeSelectionRegisterOptions options) {
  return std::make_unique<CoralNPUTileSizeSelectionRegisterPass>(
      std::move(options));
}

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

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createCoralNPUTileSizeSelectionWorkgroupPass() {
  return std::make_unique<CoralNPUTileSizeSelectionWorkgroupPass>();
}

}  // namespace mlir::coralnpu_compiler
