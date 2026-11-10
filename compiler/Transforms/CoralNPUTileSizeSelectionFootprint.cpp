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

#include "compiler/Transforms/CoralNPUTileSizeSelectionFootprint.h"

#include "compiler/Transforms/CoralNPUTileSizeSelectionUtils.h"

// IREE:
#include "iree/compiler/Codegen/Utils/Utils.h"

// MLIR:
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Remarks.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"

// LLVM:
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

// Standard library:
#include <algorithm>
#include <optional>
#include <string>
#include <utility>

#define DEBUG_TYPE "coralnpu-tile-size-selection"

using namespace mlir;
using namespace mlir::iree_compiler;

namespace mlir::coralnpu_compiler {

namespace {

// Each DTCM buffer is rounded up to this alignment.
constexpr int64_t kBufferAlignmentBytes = 64;

// Returns the size in bits of an element of `type`. Types that are not
// integers or floats (e.g. index or complex) use 64 bits as an upper bound.
int64_t getElementBits(Type type) {
  Type elementType = getElementTypeOrSelf(type);
  if (elementType.isIntOrFloat()) return elementType.getIntOrFloatBitWidth();
  return 64;
}

// Returns the bytes of a DTCM buffer that holds `numElements` elements of
// `type`, rounded up to the buffer alignment.
int64_t getBufferBytes(Type type, int64_t numElements) {
  int64_t bytes = llvm::divideCeil(numElements * getElementBits(type), 8);
  return llvm::alignTo(bytes, kBufferAlignmentBytes);
}

// Returns, for each dimension of `operand` of `linalgOp`, the number of indices
// that the iterations in `loopRanges` access. The span comes from the bounds of
// the indexing expression, so that e.g. the input of a convolution includes the
// halo (`d_spatial + d_filter * dilation`). If the bounds are unknown, the span
// is the full dimension size.
SmallVector<int64_t> getOperandSpans(linalg::LinalgOp linalgOp,
                                     OpOperand *operand,
                                     ArrayRef<int64_t> loopRanges) {
  auto shapedType = cast<ShapedType>(operand->get().getType());
  AffineMap map = linalgOp.getMatchingIndexingMap(operand);
  // Each loop iterates over [0, range - 1]; symbols have no known bounds.
  SmallVector<std::optional<int64_t>> lowerBounds(loopRanges.size(), 0);
  SmallVector<std::optional<int64_t>> upperBounds;
  for (int64_t range : loopRanges) upperBounds.push_back(range - 1);
  lowerBounds.append(map.getNumSymbols(), std::nullopt);
  upperBounds.append(map.getNumSymbols(), std::nullopt);

  SmallVector<int64_t> spans;
  for (auto [dim, expr] : llvm::enumerate(map.getResults())) {
    std::optional<int64_t> lb =
        getBoundForAffineExpr(expr, map.getNumDims(), map.getNumSymbols(),
                              lowerBounds, upperBounds, /*isUpper=*/false);
    std::optional<int64_t> ub =
        getBoundForAffineExpr(expr, map.getNumDims(), map.getNumSymbols(),
                              lowerBounds, upperBounds, /*isUpper=*/true);
    // Use the span of the accessed indices when the bounds are known.
    if (lb && ub) {
      spans.push_back(*ub - *lb + 1);
      continue;
    }

    int64_t dimSize = shapedType.getDimSize(dim);
    std::string exprStr;
    llvm::raw_string_ostream os(exprStr);
    os << expr;
    LDBG() << "could not estimate bound for dimension " << dim
           << " of operand #" << operand->getOperandNumber()
           << " (expression: " << os.str() << ") of '"
           << linalgOp->getName().getStringRef()
           << "', falling back to full dimension size (" << dimSize << ")";
    remark::missed(
        linalgOp.getLoc(),
        remark::RemarkOpts::name("DTCMFootprint").category(DEBUG_TYPE))
        << remark::reason(
               "could not estimate bound for dimension {0} of operand #{1} "
               "(expression: {2}) of '{3}', falling back to full dimension "
               "size ({4})",
               dim, operand->getOperandNumber(), os.str(),
               linalgOp->getName().getStringRef(), dimSize);
    // A dynamic dimension has no size to fall back to; count one index.
    spans.push_back(ShapedType::isDynamic(dimSize) ? 1 : dimSize);
  }
  return spans;
}

// Returns the loop ranges of `linalgOp` in a tile where the operand with the
// indexing map `map` has the sizes `spans`. A loop that `map` uses directly as
// a dimension gets the span of that dimension. The other loops keep their full
// static range.
SmallVector<int64_t> getLoopRangesFromSpans(linalg::LinalgOp linalgOp,
                                            AffineMap map,
                                            ArrayRef<int64_t> spans) {
  SmallVector<int64_t> loopRanges = linalgOp.getStaticLoopRanges();
  for (auto [expr, span] : llvm::zip_equal(map.getResults(), spans)) {
    auto dimExpr = dyn_cast<AffineDimExpr>(expr);
    // Keep the full range of loops in compound expressions (e.g. d0 + d1).
    if (!dimExpr) continue;
    int64_t &range = loopRanges[dimExpr.getPosition()];
    range = ShapedType::isDynamic(range) ? span : std::min(range, span);
  }
  // A loop with a dynamic range that no dimension sets counts one iteration.
  for (int64_t &range : loopRanges) {
    if (ShapedType::isDynamic(range)) range = 1;
  }
  return loopRanges;
}

// Visits the root op and the compute ops fused with it, and adds up the DTCM
// buffers of one tile. Only the DPS inputs of the ops are input slices: the
// DPS inits are the output staging buffers and the accumulator. The walk goes
// from the root op to the fused producers (of the inputs and of the inits) and
// to the fused consumers, and from each visited op on in both directions.
class FootprintWalker {
 public:
  explicit FootprintWalker(const llvm::SmallPtrSetImpl<Operation *> &computeOps)
      : computeOps(computeOps) {}

  // Visits the root op, with the loop ranges `loopRanges` of its DTCM tile, and
  // the fused ops reachable from it. `splitsReduction` is true if the tile
  // splits a reduction loop.
  void visitRoot(linalg::LinalgOp rootOp, ArrayRef<int64_t> loopRanges,
                 bool splitsReduction) {
    visited.insert(rootOp);
    worklist.push_back({rootOp, SmallVector<int64_t>(loopRanges),
                        /*needsAccumulator=*/splitsReduction});
    while (!worklist.empty()) {
      WorkItem item = worklist.pop_back_val();
      visitInputs(item.op, item.loopRanges);
      visitResults(item.op, item.loopRanges, item.needsAccumulator);
    }
  }

  // Returns the total bytes: the input slices times `buffersPerInput`, plus the
  // output staging buffers and the accumulator.
  int64_t getTotalBytes(int64_t buffersPerInput) const {
    int64_t inputBytes = 0;
    for (const auto &[value, bytes] : inputSliceBytes) inputBytes += bytes;
    return inputBytes * buffersPerInput + outputBytes;
  }

 private:
  struct WorkItem {
    linalg::LinalgOp op;
    SmallVector<int64_t> loopRanges;
    bool needsAccumulator;
  };

  // Returns true if `op` is a linalg op that is fused with the root op.
  bool isFusedLinalgOp(Operation *op) const {
    return isa<linalg::LinalgOp>(op) && computeOps.contains(op);
  }

  // Adds the input slice of `value`. If the fused ops read `value` more than
  // once, they share one buffer for the largest slice.
  void addInputSlice(Value value, int64_t bytes) {
    int64_t &sliceBytes = inputSliceBytes[value];
    sliceBytes = std::max(sliceBytes, bytes);
  }

  // If a fused linalg op produces `value`, enqueues that producer for the slice
  // of `value` with the sizes `spans` (once for each producer), and returns
  // true. Returns false if no fused linalg op produces `value`. Visiting the
  // producer adds its input slices, and its results like those of a consumer
  // (e.g. the output staging buffer of a producer result that the dispatch
  // also stores, and the other fused consumers of the producer).
  bool enqueueFusedProducer(Value value, ArrayRef<int64_t> spans) {
    Operation *producer = value.getDefiningOp();
    if (!producer || !isFusedLinalgOp(producer)) return false;
    // Visit each producer once (e.g. a producer with 2 users).
    // TODO(sflur): what if each user uses a different slice of the producer?
    // Check how the fusion is done and match the counting here.
    if (!visited.insert(producer).second) return true;
    auto producerOp = cast<linalg::LinalgOp>(producer);
    AffineMap resultMap =
        producerOp.getIndexingMapMatchingResult(cast<OpResult>(value));
    worklist.push_back({producerOp,
                        getLoopRangesFromSpans(producerOp, resultMap, spans),
                        /*needsAccumulator=*/false});
    return true;
  }

  // Adds the input slices of the DPS inputs of `op`, which iterates over
  // `loopRanges`.
  void visitInputs(linalg::LinalgOp op, ArrayRef<int64_t> loopRanges) {
    for (OpOperand *operand : op.getDpsInputOperands()) {
      Value value = operand->get();
      // Scalars live in registers.
      if (!isa<ShapedType>(value.getType())) continue;
      SmallVector<int64_t> spans = getOperandSpans(op, operand, loopRanges);
      // A tensor that a fused linalg op produces needs no buffer: the producer
      // computes it from its own inputs, so visit these inputs instead.
      if (enqueueFusedProducer(value, spans)) continue;
      // Empty tensors and constants are not copied to DTCM.
      if (isa_and_nonnull<tensor::EmptyOp>(value.getDefiningOp()) ||
          matchPattern(value, m_Constant())) {
        continue;
      }
      int64_t numElements = llvm::product_of(spans);
      addInputSlice(value, getBufferBytes(value.getType(), numElements));
    }
  }

  // Adds the output buffers of the results of `op`, which iterates over
  // `loopRanges`, and enqueues the fused producers of the inits and the fused
  // consumers of these results. `needsAccumulator` is true if each result
  // needs a DTCM buffer even if it does not leave the fused ops (the root
  // result with split K).
  void visitResults(linalg::LinalgOp op, ArrayRef<int64_t> loopRanges,
                    bool needsAccumulator) {
    for (OpResult result : op->getResults()) {
      OpOperand *init = op.getDpsInitOperand(result.getResultNumber());
      SmallVector<int64_t> spans = getOperandSpans(op, init, loopRanges);
      // A fused op that computes the init (e.g. a broadcast of a bias) reads
      // its inputs in the tile too. A fill reads only a scalar.
      enqueueFusedProducer(init->get(), spans);
      // A result that leaves the fused ops (e.g. that is stored) needs an
      // output staging buffer, which also holds the accumulator. Else, only
      // the accumulator needs a buffer.
      bool leavesFusedOps = llvm::any_of(
          result.getUsers(), [&](Operation *u) { return !isFusedLinalgOp(u); });
      if (leavesFusedOps || needsAccumulator) {
        outputBytes +=
            getBufferBytes(result.getType(), llvm::product_of(spans));
      }

      // Enqueue each fused consumer once, with the tile of this result.
      for (OpOperand &use : result.getUses()) {
        Operation *user = use.getOwner();
        if (!isFusedLinalgOp(user) || !visited.insert(user).second) continue;
        auto consumer = cast<linalg::LinalgOp>(user);
        worklist.push_back(
            {consumer,
             getLoopRangesFromSpans(
                 consumer, consumer.getMatchingIndexingMap(&use), spans),
             /*needsAccumulator=*/false});
      }
    }
  }

  const llvm::SmallPtrSetImpl<Operation *> &computeOps;
  llvm::SmallPtrSet<Operation *, 8> visited;
  SmallVector<WorkItem, 8> worklist;
  // The bytes of the input slice of each tensor that is read from memory.
  llvm::SmallDenseMap<Value, int64_t> inputSliceBytes;
  // The bytes of the output staging buffers and of the accumulator.
  int64_t outputBytes = 0;
};

// Helper to extract constant inner tile size for a given dimension if present.
int64_t getInnerTileSize(ArrayRef<int64_t> innerDimsPos,
                         ArrayRef<OpFoldResult> mixedTiles, int64_t dim) {
  auto it = llvm::find(innerDimsPos, dim);
  if (it == innerDimsPos.end()) return 1;
  size_t k = std::distance(innerDimsPos.begin(), it);
  return getConstantIntValue(mixedTiles[k]).value_or(1);
}

// Estimates the footprint in bytes of a packed tensor operand (rank N + M).
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

  return llvm::divideCeil(numElements * getElementBits(type), 8);
}

// Estimates the footprint in bytes of an unpacked tensor operand (rank N).
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

  return llvm::divideCeil(numElements * getElementBits(type), 8);
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

  return llvm::divideCeil(
      rankedTensorType.getNumElements() * getElementBits(rankedTensorType), 8);
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

  return llvm::divideCeil(
      rankedTensorType.getNumElements() * getElementBits(rankedTensorType), 8);
}

// Estimates the tiled footprint of a single tensor operand of an op that is not
// a linalg op.
int64_t estimateOperandFootprint(Operation *op, OpOperand *operand,
                                 ArrayRef<int64_t> tileSizes) {
  Value value = operand->get();

  if (auto packOp = dyn_cast<linalg::PackOp>(op)) {
    return estimatePackOperandFootprint(packOp, operand, tileSizes);
  }
  if (auto unpackOp = dyn_cast<linalg::UnPackOp>(op)) {
    return estimateUnpackOperandFootprint(unpackOp, operand, tileSizes);
  }

  // Fallback for other operations
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

  return llvm::divideCeil(numElements * getElementBits(rankedTensorType), 8);
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

  int64_t elementBits = getElementBits(destType);

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
    return llvm::divideCeil(2 * totalElements * elementBits, 8);
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

  int64_t buffer1Elements = innerM * (sizeN + 2);
  int64_t buffer2Elements = (sizeM + 2) * (sizeN + paddingN);

  return llvm::divideCeil((buffer1Elements + 2 * buffer2Elements) * elementBits,
                          8);
}

// Estimates the footprint of a root op that is not a linalg op (e.g. pack or
// unpack) from its own tensor operands: the inputs times `buffersPerInput`,
// the inits, and the temporaries of an unpack, which depend on the vector tile
// sizes `vectorParallelSizes`.
int64_t estimateNonLinalgRootFootprint(Operation *rootOp,
                                       ArrayRef<int64_t> tileSizes,
                                       ArrayRef<int64_t> vectorParallelSizes,
                                       int64_t buffersPerInput) {
  auto dpsOp = dyn_cast<DestinationStyleOpInterface>(rootOp);
  int64_t totalBytes = 0;
  for (OpOperand &operand : rootOp->getOpOperands()) {
    // Scalars live in registers.
    // TODO(sflur): not always; we should count them in totalBytes.
    if (!isa<ShapedType>(operand.get().getType())) continue;
    int64_t bytes =
        llvm::alignTo(estimateOperandFootprint(rootOp, &operand, tileSizes),
                      kBufferAlignmentBytes);
    // An init is an output buffer; an input has `buffersPerInput` buffers.
    bool isInit = dpsOp && dpsOp.isDpsInit(&operand);
    totalBytes += isInit ? bytes : bytes * buffersPerInput;
  }

  // The lowering of an unpack also needs temporary buffers.
  if (auto unpackOp = dyn_cast<linalg::UnPackOp>(rootOp)) {
    totalBytes += llvm::alignTo(estimateUnpackTemporariesFootprint(
                                    unpackOp, tileSizes, vectorParallelSizes),
                                kBufferAlignmentBytes);
  }
  return totalBytes;
}

// Returns the extents of the vector tiles of a loop with the full range
// `range`, the DTCM tile size `tile` and the vector tile size `vector` (0 for a
// loop without a vector tile): the vector tile, and the remainders that loop
// peeling gives in a DTCM tile and in the last DTCM tile, if it is partial.
SmallVector<int64_t> getVectorTileExtents(int64_t range, int64_t tile,
                                          int64_t vector) {
  if (range <= 0) return {};
  if (tile <= 0 || tile > range) tile = range;
  SmallVector<int64_t> extents;
  // Adds the extents of the vector tiles of a DTCM tile of `tileExtent`.
  auto addDTCMTile = [&](int64_t tileExtent) {
    int64_t extent = vector > 0 ? std::min(vector, tileExtent) : tileExtent;
    extents.push_back(extent);
    // Loop peeling gives a remainder vector tile.
    if (tileExtent % extent != 0) extents.push_back(tileExtent % extent);
  };
  addDTCMTile(tile);
  // The last DTCM tile is partial if the tile size does not divide the range.
  if (range % tile != 0) addDTCMTile(range % tile);
  llvm::sort(extents);
  extents.erase(std::unique(extents.begin(), extents.end()), extents.end());
  return extents;
}

// Returns true if `op` only narrows its input from f32 to bf16 with one
// arith.truncf (e.g. the linalg.copy of CoralNPUPromoteBF16Accumulator), and no
// op of `computeOps` reads its result.
bool isStoredBF16Truncation(
    Operation *op, const llvm::SmallPtrSetImpl<Operation *> &computeOps) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp || linalgOp.getNumDpsInputs() != 1 ||
      op->getNumResults() != 1 || !linalg::isElementwise(linalgOp) ||
      !getElementTypeOrSelf(linalgOp.getDpsInputOperand(0)->get()).isF32() ||
      !getElementTypeOrSelf(op->getResult(0)).isBF16()) {
    return false;
  }
  // The body is one truncf (with the default rounding) of the input.
  Block *body = linalgOp.getBlock();
  auto yieldOp = dyn_cast<linalg::YieldOp>(body->getTerminator());
  auto truncOp = yieldOp && yieldOp.getNumOperands() == 1
                     ? yieldOp.getOperand(0).getDefiningOp<arith::TruncFOp>()
                     : arith::TruncFOp();
  if (!truncOp || truncOp.getRoundingmode() ||
      truncOp.getIn() != body->getArgument(0) ||
      !llvm::hasSingleElement(body->without_terminator())) {
    return false;
  }
  // The narrowed result is stored, not read by a fused op.
  return llvm::none_of(op->getUsers(), [&](Operation *user) {
    return isa<linalg::LinalgOp>(user) && computeOps.contains(user);
  });
}

// Returns the bytes of the scratch tiles that CoralNPUMatrixCodegen allocates
// in DTCM for the root op `rootOp`, in a DTCM tile of the root loops with the
// ranges `tileRanges`. `loopRanges` are the full static loop ranges.
//
// The Zvt asm takes a vector tile of a contraction, or of a convolution with
// the Zvt layout (isZvtMatrixContractionShape), if the target has +zvtbase,
// the asm can load both operands with Zvt element types (isZvtReadableInputs),
// the init is zero, the shape of the accumulator is a Zvt shape
// (isZvtAccumulatorShape), and K is 1 in the vector tile, in a K loop. If a
// fused op reads the root result, the asm cannot drain the accumulators
// straight to a stored tile: it drains them to a scratch tile of
// alignTo(M, 16) x N accumulator elements, one for each distinct shape (the
// vector tile and the remainders of loop peeling), and the fused ops read the
// scratch tile. With +zvfbfmin, a root result that a fused op only narrows to
// bf16 drains straight to the stored tile if the rows are a multiple of 4.
//
// The model does not count:
// - the scratch tiles of a root whose DTCM tile splits K: the DTCM copies in
//   the K loop keep the asm from taking the root;
// - the scratch tile (16x32xf32) of the BF16 vecmat lowering of the same pass,
//   which takes only some of the K loops that the Zvt asm does not take.
// The model counts the scratch tile of a 1-D accumulator (e.g. a vecmat), but
// today DropVectorUnitDims makes such a contraction a vector.fma before the
// asm, so this count is conservative.
//
// Keep in sync with CoralNPUMatrixCodegen.cpp.
int64_t estimateZvtScratchBytes(
    linalg::LinalgOp rootOp, ArrayRef<int64_t> loopRanges,
    ArrayRef<int64_t> tileRanges, bool splitsReduction,
    ArrayRef<int64_t> vectorParallelSizes,
    ArrayRef<int64_t> vectorReductionSizes,
    const llvm::SmallPtrSetImpl<Operation *> &computeOps) {
  int64_t numLoops = loopRanges.size();
  // The asm needs +zvtbase, and a contraction or a convolution that the conv
  // vectorizer lowers to contractions with the Zvt layout.
  bool isZvtConvolution =
      linalg::isaConvolutionOpInterface(rootOp) &&
      isZvtMatrixContractionShape(rootOp.getOperation(),
                                  /*strictLayoutAndSingleUse=*/false);
  if (!hasZvtTargetFeature(rootOp) || splitsReduction ||
      vectorParallelSizes.size() != numLoops ||
      vectorReductionSizes.size() != numLoops ||
      !(linalg::isaContractionOpInterface(rootOp) || isZvtConvolution) ||
      rootOp.getNumDpsInputs() != 2 || rootOp.getNumDpsInits() != 1) {
    return 0;
  }
  OpOperand *lhs = rootOp.getDpsInputOperand(0);
  OpOperand *rhs = rootOp.getDpsInputOperand(1);
  OpOperand *init = rootOp.getDpsInitOperand(0);
  Type accType = getElementTypeOrSelf(init->get());
  // The asm loads its operands, with Zvt element types (isZvtReadableInputs):
  // it does not take an operand that a fused op computes (e.g. a
  // dequantization), though RVV tiling can clone multi-use or non-1:1
  // extensions into a K loop that the asm still lowers.
  if (!isZvtReadableInputs(rootOp, /*strictLayoutAndSingleUse=*/false))
    return 0;

  auto isFusedLinalgOp = [&](Operation *op) {
    return isa_and_nonnull<linalg::LinalgOp>(op) && computeOps.contains(op);
  };
  // The asm zeroes the accumulators (or loads them): a computed init keeps the
  // root on RVV.
  if (Operation *initProducer = init->get().getDefiningOp();
      isFusedLinalgOp(initProducer)) {
    auto fillOp = dyn_cast<linalg::FillOp>(initProducer);
    if (!fillOp || !(matchPattern(fillOp.getInputs()[0], m_Zero()) ||
                     matchPattern(fillOp.getInputs()[0], m_AnyZeroFloat()))) {
      return 0;
    }
  }
  // Without a fused op that reads the root result, the asm drains the
  // accumulators straight to the stored tile.
  SmallVector<Operation *> readers;
  for (Operation *user : rootOp->getUsers()) {
    if (isFusedLinalgOp(user)) readers.push_back(user);
  }
  if (readers.empty()) return 0;
  bool narrowsToBF16 = readers.size() == 1 &&
                       isStoredBF16Truncation(readers.front(), computeOps) &&
                       hasCoralNPUTargetFeature(rootOp, "+zvfbfmin");

  // Each vector tile has a K of 1. The tile range of K is the trip count of
  // the K loops.
  int64_t kTripCount = 1;
  for (auto [loop, iteratorType] :
       llvm::enumerate(rootOp.getIteratorTypesArray())) {
    if (iteratorType != utils::IteratorType::reduction) continue;
    if (vectorReductionSizes[loop] != 1 && tileRanges[loop] != 1) return 0;
    kTripCount *= tileRanges[loop];
  }

  // The loops of the accumulator dims, and the extents of their vector tiles.
  AffineMap lhsMap = rootOp.getMatchingIndexingMap(lhs);
  AffineMap rhsMap = rootOp.getMatchingIndexingMap(rhs);
  AffineMap initMap = rootOp.getMatchingIndexingMap(init);
  SmallVector<unsigned> accLoops;
  SmallVector<SmallVector<int64_t>> accExtents;
  for (AffineExpr expr : initMap.getResults()) {
    auto dimExpr = dyn_cast<AffineDimExpr>(expr);
    // The vector tiles of other indexing expressions are not modeled.
    if (!dimExpr) return 0;
    unsigned loop = dimExpr.getPosition();
    accLoops.push_back(loop);
    accExtents.push_back(getVectorTileExtents(
        loopRanges[loop], tileRanges[loop], vectorParallelSizes[loop]));
  }

  // The (rows, columns) of the distinct scratch tiles.
  llvm::SmallSetVector<std::pair<int64_t, int64_t>, 4> scratchTiles;
  // Adds the scratch tile of the vector tiles with the accumulator extents
  // `extents` (one for each accumulator dim), if the asm takes them.
  auto addScratchTile = [&](ArrayRef<int64_t> extents) {
    // The loops and the extents of the dims that remain after the unit dims
    // are dropped.
    SmallVector<unsigned> loops;
    SmallVector<int64_t> sizes;
    for (auto [loop, extent] : llvm::zip_equal(accLoops, extents)) {
      if (extent == 1) continue;
      loops.push_back(loop);
      sizes.push_back(extent);
    }
    // The asm takes 1-D and 2-D accumulators.
    if (loops.empty() || loops.size() > 2) return;
    bool rank1 = loops.size() == 1;
    unsigned rowLoop = loops.front(), colLoop = loops.back();
    int64_t m = rank1 ? 1 : sizes.front(), n = sizes.back();
    bool swap = !rank1 && rhsMap.isFunctionOfDim(rowLoop);
    // The left-hand side indexes the rows and the right-hand side the columns
    // (the opposite with the swap). A 1-D accumulator is along N.
    bool layoutMatches = rank1 ? !lhsMap.isFunctionOfDim(colLoop) &&
                                     rhsMap.isFunctionOfDim(colLoop)
                               : (swap ? lhsMap.isFunctionOfDim(colLoop)
                                       : lhsMap.isFunctionOfDim(rowLoop) &&
                                             rhsMap.isFunctionOfDim(colLoop));
    // Without a K loop, the asm takes only full 16x16 tiles.
    bool hasKLoopOrFullTiles =
        kTripCount > 1 || (!rank1 && m % 16 == 0 && n % 16 == 0);
    // A drain that narrows to bf16 writes 4 rows at a time.
    bool drainsToStore = narrowsToBF16 && m % 4 == 0;
    if (!layoutMatches || !hasKLoopOrFullTiles || drainsToStore ||
        !isZvtAccumulatorShape(m, n, rank1, swap)) {
      return;
    }
    scratchTiles.insert({rank1 ? 16 : llvm::alignTo(m, 16), n});
  };

  // Visit each combination of the extents of the accumulator dims.
  SmallVector<size_t> choice(accLoops.size(), 0);
  SmallVector<int64_t> extents(accLoops.size());
  while (true) {
    for (size_t d = 0; d < accLoops.size(); ++d) {
      extents[d] = accExtents[d][choice[d]];
    }
    addScratchTile(extents);
    // Go to the next combination, or stop after the last one.
    size_t d = 0;
    for (; d < choice.size(); ++d) {
      if (++choice[d] < accExtents[d].size()) break;
      choice[d] = 0;
    }
    if (d == choice.size()) break;
  }

  int64_t totalBytes = 0;
  for (auto [rows, cols] : scratchTiles) {
    totalBytes += getBufferBytes(accType, rows * cols);
  }
  return totalBytes;
}

}  // namespace

DTCMFootprintModel::DTCMFootprintModel(Operation *rootOp,
                                       ArrayRef<Operation *> computeOps,
                                       int64_t buffersPerInput)
    : rootOp(rootOp),
      computeOps(computeOps.begin(), computeOps.end()),
      buffersPerInput(buffersPerInput) {}

int64_t DTCMFootprintModel::estimate(ArrayRef<int64_t> tileSizes,
                                     ArrayRef<int64_t> vectorParallelSizes,
                                     ArrayRef<int64_t> vectorReductionSizes,
                                     bool streamReduction) const {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(rootOp);
  // A root that is not a linalg op uses the model of its own operands.
  if (!linalgOp) {
    return estimateNonLinalgRootFootprint(rootOp, tileSizes,
                                          vectorParallelSizes, buffersPerInput);
  }

  // The loop ranges of the tile, and whether the tile splits a reduction loop.
  SmallVector<int64_t> staticRanges = linalgOp.getStaticLoopRanges();
  SmallVector<int64_t> loopRanges(staticRanges);
  SmallVector<utils::IteratorType> iteratorTypes =
      linalgOp.getIteratorTypesArray();
  bool splitsReduction = false;
  for (auto [range, tileSize, iteratorType] :
       llvm::zip_equal(loopRanges, tileSizes, iteratorTypes)) {
    // A tile size of 0, or of the full range, does not tile the loop.
    if (tileSize <= 0 || tileSize >= range) continue;
    range = tileSize;
    splitsReduction |= iteratorType == utils::IteratorType::reduction;
  }

  // The loop ranges for the input slices: a streamed reduction loop reads one
  // vector step of the inputs at a time (all of it without a vector tile).
  SmallVector<int64_t> inputRanges(loopRanges);
  if (streamReduction) {
    for (auto [range, vectorSize, iteratorType] :
         llvm::zip_equal(inputRanges, vectorReductionSizes, iteratorTypes)) {
      if (iteratorType != utils::IteratorType::reduction || vectorSize <= 0) {
        continue;
      }
      range = std::min(range, vectorSize);
    }
  }

  // The results do not depend on the reduction loops, so the walker can use
  // the input ranges for the whole tile.
  FootprintWalker walker(computeOps);
  walker.visitRoot(linalgOp, inputRanges, splitsReduction);
  return walker.getTotalBytes(buffersPerInput) +
         estimateZvtScratchBytes(linalgOp, staticRanges, loopRanges,
                                 splitsReduction, vectorParallelSizes,
                                 vectorReductionSizes, computeOps);
}

}  // namespace mlir::coralnpu_compiler
