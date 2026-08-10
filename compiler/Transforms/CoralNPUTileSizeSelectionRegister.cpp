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
// CoralNPU Register Tile Size Selection Pass
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <memory>
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
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

// LLVM headers
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"

#define DEBUG_TYPE "coralnpu-tile-size-selection"

using namespace mlir;
using namespace mlir::iree_compiler;

namespace mlir::coralnpu_compiler {

#define GEN_PASS_DEF_CORALNPUTILESIZESELECTIONREGISTER
#include "compiler/Transforms/Passes.h.inc"

namespace {

// tiling loops: [N, OH, OW, OC, KH, KW, IC]
void setConv2DNhwcHwcfVectorSizes(
    linalg::LinalgOp op, int64_t vectorWidth,
    const CoralNPUTileSizeSelectionAnalysis &analysis,
    MutableArrayRef<int64_t> vectorParallelSizes,
    IREE::Codegen::DispatchLoweringPassPipeline &pipeline) {
  pipeline = IREE::Codegen::DispatchLoweringPassPipeline::
      CPUConvTileAndDecomposeExpert;

  const auto &loops = analysis.parallelLoops;

  int64_t ocTile = 8 * vectorWidth;
  int64_t owTile =
      analysis.staticLoopRanges[loops[0]] % vectorWidth == 0 ? 4 : 1;

  vectorParallelSizes[loops[0]] = ocTile;
  vectorParallelSizes[loops[1]] = owTile;
}

// tiling loops: [N, OH, OW, C, KH, KW]
void setPoolingNhwcVectorSizes(
    linalg::LinalgOp op, int64_t vectorWidth,
    const CoralNPUTileSizeSelectionAnalysis &analysis,
    MutableArrayRef<int64_t> vectorParallelSizes,
    IREE::Codegen::DispatchLoweringPassPipeline &pipeline) {
  pipeline = IREE::Codegen::DispatchLoweringPassPipeline::
      CPUConvTileAndDecomposeExpert;

  const auto &loops = analysis.parallelLoops;

  int64_t cTile = 8 * vectorWidth;
  int64_t owTile =
      analysis.staticLoopRanges[loops[0]] % vectorWidth == 0 ? 4 : 1;

  vectorParallelSizes[loops[0]] = cTile;
  vectorParallelSizes[loops[1]] = owTile;
}

// tiling loops: [N, OH, OW, C, KH, KW]
void setDepthwiseConv2DNhwcHwcVectorSizes(
    linalg::LinalgOp op, int64_t vectorWidth,
    const CoralNPUTileSizeSelectionAnalysis &analysis,
    MutableArrayRef<int64_t> vectorParallelSizes,
    IREE::Codegen::DispatchLoweringPassPipeline &pipeline) {
  pipeline = IREE::Codegen::DispatchLoweringPassPipeline::
      CPUConvTileAndDecomposeExpert;

  const auto &loops = analysis.parallelLoops;

  int64_t cTile = 8 * vectorWidth;
  int64_t owTile =
      analysis.staticLoopRanges[loops[0]] % vectorWidth == 0 ? 4 : 1;

  vectorParallelSizes[loops[0]] = cTile;
  vectorParallelSizes[loops[1]] = owTile;
}

// tiling loops: [[N,] OW, OC, KW, IC]
void setConv1DNwcWcfVectorSizes(
    linalg::LinalgOp op, int64_t vectorWidth,
    const CoralNPUTileSizeSelectionAnalysis &analysis,
    MutableArrayRef<int64_t> vectorParallelSizes,
    IREE::Codegen::DispatchLoweringPassPipeline &pipeline) {
  pipeline = IREE::Codegen::DispatchLoweringPassPipeline::
      CPUConvTileAndDecomposeExpert;

  const auto &loops = analysis.parallelLoops;

  int64_t ocTile = 8 * vectorWidth;
  int64_t owTile =
      analysis.staticLoopRanges[loops[0]] % vectorWidth == 0 ? 4 : 1;

  vectorParallelSizes[loops[0]] = ocTile;
  vectorParallelSizes[loops[1]] = owTile;
}

// tiling loops: [[N,] OC, OW, IC, KW]
void setConv1DNcwFcwVectorSizes(
    linalg::LinalgOp op, int64_t vectorWidth,
    const CoralNPUTileSizeSelectionAnalysis &analysis,
    MutableArrayRef<int64_t> vectorParallelSizes,
    IREE::Codegen::DispatchLoweringPassPipeline &pipeline) {
  pipeline = IREE::Codegen::DispatchLoweringPassPipeline::CPUDoubleTilingExpert;

  const auto &loops = analysis.parallelLoops;

  int64_t owTile = vectorWidth;
  int64_t ocTile =
      analysis.staticLoopRanges[loops[0]] % vectorWidth == 0 ? 4 : 1;

  vectorParallelSizes[loops[0]] = owTile;
  vectorParallelSizes[loops[1]] = ocTile;
}

// tiling loops: [[N,] OC, OH, OW, IC, KH, KW]
void setConv2DNchwFchwVectorSizes(
    linalg::LinalgOp op, int64_t vectorWidth,
    const CoralNPUTileSizeSelectionAnalysis &analysis,
    MutableArrayRef<int64_t> vectorParallelSizes,
    IREE::Codegen::DispatchLoweringPassPipeline &pipeline) {
  pipeline = IREE::Codegen::DispatchLoweringPassPipeline::CPUDoubleTilingExpert;

  const auto &loops = analysis.parallelLoops;

  int64_t owTile = 4 * vectorWidth;

  vectorParallelSizes[loops[0]] = owTile;
}

// tiling loops: [[N,] OH, OW, OC, KH, KW, IC]
void setConv2DNhwcFhwcVectorSizes(
    linalg::LinalgOp op, int64_t vectorWidth,
    const CoralNPUTileSizeSelectionAnalysis &analysis,
    MutableArrayRef<int64_t> vectorParallelSizes,
    IREE::Codegen::DispatchLoweringPassPipeline &pipeline) {
  pipeline = IREE::Codegen::DispatchLoweringPassPipeline::CPUDoubleTilingExpert;

  const auto &loops = analysis.parallelLoops;

  int64_t ocTile = 4 * vectorWidth;
  int64_t owTile =
      analysis.staticLoopRanges[loops[0]] % vectorWidth == 0 ? 4 : 1;

  vectorParallelSizes[loops[0]] = ocTile;
  vectorParallelSizes[loops[1]] = owTile;
}

// tiling loops: [[N,] OH, OW, G, OC, KH, KW, C]
void setConv2DNhwgcGfhwcVectorSizes(
    linalg::LinalgOp op, int64_t vectorWidth,
    const CoralNPUTileSizeSelectionAnalysis &analysis,
    MutableArrayRef<int64_t> vectorParallelSizes,
    IREE::Codegen::DispatchLoweringPassPipeline &pipeline) {
  pipeline = IREE::Codegen::DispatchLoweringPassPipeline::
      CPUConvTileAndDecomposeExpert;

  const auto &loops = analysis.parallelLoops;

  int64_t ocTile = 8 * vectorWidth;

  vectorParallelSizes[loops[0]] = ocTile;
}

// tiling loops: [[N,] G, OC, OH, OW, C, KH, KW]
void setConv2DNgchwGfchwVectorSizes(
    linalg::LinalgOp op, int64_t vectorWidth,
    const CoralNPUTileSizeSelectionAnalysis &analysis,
    MutableArrayRef<int64_t> vectorParallelSizes,
    IREE::Codegen::DispatchLoweringPassPipeline &pipeline) {
  pipeline = IREE::Codegen::DispatchLoweringPassPipeline::CPUDoubleTilingExpert;

  const auto &loops = analysis.parallelLoops;

  int64_t owTile = 8 * vectorWidth;

  vectorParallelSizes[loops[0]] = owTile;
}

// tiling loops: [OD, OH, OW, KD, KH, KW]
void setConv3DVectorSizes(
    linalg::LinalgOp op, int64_t vectorWidth,
    const CoralNPUTileSizeSelectionAnalysis &analysis,
    MutableArrayRef<int64_t> vectorParallelSizes,
    IREE::Codegen::DispatchLoweringPassPipeline &pipeline) {
  pipeline = IREE::Codegen::DispatchLoweringPassPipeline::CPUDoubleTilingExpert;

  const auto &loops = analysis.parallelLoops;

  int64_t owTile = 8 * vectorWidth;

  vectorParallelSizes[loops[0]] = owTile;
}

// tiling loops: [[N,] OD, OH, OW, OC, KD, KH, KW, IC]
void setConv3DNdhwcDhwcfVectorSizes(
    linalg::LinalgOp op, int64_t vectorWidth,
    const CoralNPUTileSizeSelectionAnalysis &analysis,
    MutableArrayRef<int64_t> vectorParallelSizes,
    IREE::Codegen::DispatchLoweringPassPipeline &pipeline) {
  pipeline = IREE::Codegen::DispatchLoweringPassPipeline::
      CPUConvTileAndDecomposeExpert;

  const auto &loops = analysis.parallelLoops;

  int64_t ocTile = 4 * vectorWidth;
  int64_t owTile =
      analysis.staticLoopRanges[loops[0]] % vectorWidth == 0 ? 4 : 1;

  vectorParallelSizes[loops[0]] = ocTile;
  vectorParallelSizes[loops[1]] = owTile;
}

// tiling loops: [[N,] OC, OD, OH, OW, IC, KD, KH, KW]
void setConv3DNcdhwFcdhwVectorSizes(
    linalg::LinalgOp op, int64_t vectorWidth,
    const CoralNPUTileSizeSelectionAnalysis &analysis,
    MutableArrayRef<int64_t> vectorParallelSizes,
    IREE::Codegen::DispatchLoweringPassPipeline &pipeline) {
  pipeline = IREE::Codegen::DispatchLoweringPassPipeline::CPUDoubleTilingExpert;

  const auto &loops = analysis.parallelLoops;

  int64_t ocTile = 4 * vectorWidth;
  int64_t owTile =
      analysis.staticLoopRanges[loops[0]] % vectorWidth == 0 ? 4 : 1;

  vectorParallelSizes[loops[0]] = ocTile;
  vectorParallelSizes[loops[1]] = owTile;
}

// tiling loops: [[N,] OD, OH, OW, C, KD, KH, KW]
void setDepthwiseConv3DNdhwcDhwcVectorSizes(
    linalg::LinalgOp op, int64_t vectorWidth,
    const CoralNPUTileSizeSelectionAnalysis &analysis,
    MutableArrayRef<int64_t> vectorParallelSizes,
    IREE::Codegen::DispatchLoweringPassPipeline &pipeline) {
  pipeline = IREE::Codegen::DispatchLoweringPassPipeline::
      CPUConvTileAndDecomposeExpert;

  const auto &loops = analysis.parallelLoops;

  int64_t cTile = 4 * vectorWidth;
  int64_t owTile = 2 * vectorWidth;

  vectorParallelSizes[loops[0]] = cTile;
  vectorParallelSizes[loops[1]] = owTile;
}

// tiling loops: [[N,] OD, OH, OW, C, M, KD, KH, KW]
void setDepthwiseConv3DNdhwcDhwcmVectorSizes(
    linalg::LinalgOp op, int64_t vectorWidth,
    const CoralNPUTileSizeSelectionAnalysis &analysis,
    MutableArrayRef<int64_t> vectorParallelSizes,
    IREE::Codegen::DispatchLoweringPassPipeline &pipeline) {
  pipeline = IREE::Codegen::DispatchLoweringPassPipeline::CPUDoubleTilingExpert;

  const auto &loops = analysis.parallelLoops;

  int64_t mTile = 4 * vectorWidth;
  int64_t cTile = 2 * vectorWidth;

  vectorParallelSizes[loops[0]] = mTile;
  vectorParallelSizes[loops[1]] = cTile;
}

// tiling loops: [[N,] OD, OH, OW, C, KD, KH, KW]
void setPoolingNdhwcVectorSizes(
    linalg::LinalgOp op, int64_t vectorWidth,
    const CoralNPUTileSizeSelectionAnalysis &analysis,
    MutableArrayRef<int64_t> vectorParallelSizes,
    IREE::Codegen::DispatchLoweringPassPipeline &pipeline) {
  pipeline = IREE::Codegen::DispatchLoweringPassPipeline::
      CPUConvTileAndDecomposeExpert;

  const auto &loops = analysis.parallelLoops;

  int64_t cTile = vectorWidth;
  int64_t owTile = vectorWidth;

  vectorParallelSizes[loops[0]] = cTile;
  vectorParallelSizes[loops[1]] = owTile;
}

// tiling loops: [OW, KW]
void setConv1DVectorSizes(
    linalg::LinalgOp op, int64_t vectorWidth,
    const CoralNPUTileSizeSelectionAnalysis &analysis,
    MutableArrayRef<int64_t> vectorParallelSizes,
    IREE::Codegen::DispatchLoweringPassPipeline &pipeline) {
  pipeline = IREE::Codegen::DispatchLoweringPassPipeline::CPUDoubleTilingExpert;

  const auto &loops = analysis.parallelLoops;

  int64_t owTile = 4 * vectorWidth;

  vectorParallelSizes[loops[0]] = owTile;
}

// tiling loops: [OH, OW, KH, KW]
void setConv2DVectorSizes(
    linalg::LinalgOp op, int64_t vectorWidth,
    const CoralNPUTileSizeSelectionAnalysis &analysis,
    MutableArrayRef<int64_t> vectorParallelSizes,
    IREE::Codegen::DispatchLoweringPassPipeline &pipeline) {
  pipeline = IREE::Codegen::DispatchLoweringPassPipeline::CPUDoubleTilingExpert;

  const auto &loops = analysis.parallelLoops;

  int64_t owTile = 4 * vectorWidth;

  vectorParallelSizes[loops[0]] = owTile;
}

// tiling loops: [[N,] OW, C, KW]
void setDepthwiseConv1DNwcWcVectorSizes(
    linalg::LinalgOp op, int64_t vectorWidth,
    const CoralNPUTileSizeSelectionAnalysis &analysis,
    MutableArrayRef<int64_t> vectorParallelSizes,
    IREE::Codegen::DispatchLoweringPassPipeline &pipeline) {
  pipeline = IREE::Codegen::DispatchLoweringPassPipeline::
      CPUConvTileAndDecomposeExpert;

  const auto &loops = analysis.parallelLoops;

  int64_t cTile = vectorWidth;
  int64_t owTile = vectorWidth;

  vectorParallelSizes[loops[0]] = cTile;
  vectorParallelSizes[loops[1]] = owTile;
}

// tiling loops: [[N,] C, OW, KW]
void setDepthwiseConv1DNcwCwVectorSizes(
    linalg::LinalgOp op, int64_t vectorWidth,
    const CoralNPUTileSizeSelectionAnalysis &analysis,
    MutableArrayRef<int64_t> vectorParallelSizes,
    IREE::Codegen::DispatchLoweringPassPipeline &pipeline) {
  pipeline = IREE::Codegen::DispatchLoweringPassPipeline::
      CPUConvTileAndDecomposeExpert;

  const auto &loops = analysis.parallelLoops;

  int64_t owTile = vectorWidth;
  int64_t cTile = vectorWidth;

  vectorParallelSizes[loops[0]] = owTile;
  vectorParallelSizes[loops[1]] = cTile;
}

// tiling loops: [[N,] OW, C, M, KW]
void setDepthwiseConv1DNwcWcmVectorSizes(
    linalg::LinalgOp op, int64_t vectorWidth,
    const CoralNPUTileSizeSelectionAnalysis &analysis,
    MutableArrayRef<int64_t> vectorParallelSizes,
    IREE::Codegen::DispatchLoweringPassPipeline &pipeline) {
  pipeline = IREE::Codegen::DispatchLoweringPassPipeline::CPUDoubleTilingExpert;

  const auto &loops = analysis.parallelLoops;

  int64_t mTile = vectorWidth;
  int64_t cTile = vectorWidth;

  vectorParallelSizes[loops[0]] = mTile;
  vectorParallelSizes[loops[1]] = cTile;
}

// tiling loops: [[N,] OH, OW, C, M, KH, KW]
void setDepthwiseConv2DNhwcHwcmVectorSizes(
    linalg::LinalgOp op, int64_t vectorWidth,
    const CoralNPUTileSizeSelectionAnalysis &analysis,
    MutableArrayRef<int64_t> vectorParallelSizes,
    IREE::Codegen::DispatchLoweringPassPipeline &pipeline) {
  pipeline = IREE::Codegen::DispatchLoweringPassPipeline::CPUDoubleTilingExpert;

  const auto &loops = analysis.parallelLoops;

  int64_t mTile = vectorWidth;
  int64_t cTile = vectorWidth;

  vectorParallelSizes[loops[0]] = mTile;
  vectorParallelSizes[loops[1]] = cTile;
}

// tiling loops: [[N,] OW, C, KW]
void setPoolingNwcVectorSizes(
    linalg::LinalgOp op, int64_t vectorWidth,
    const CoralNPUTileSizeSelectionAnalysis &analysis,
    MutableArrayRef<int64_t> vectorParallelSizes,
    IREE::Codegen::DispatchLoweringPassPipeline &pipeline) {
  pipeline = IREE::Codegen::DispatchLoweringPassPipeline::
      CPUConvTileAndDecomposeExpert;

  const auto &loops = analysis.parallelLoops;

  int64_t cTile = vectorWidth;
  int64_t owTile = vectorWidth;

  vectorParallelSizes[loops[0]] = cTile;
  vectorParallelSizes[loops[1]] = owTile;
}

// tiling loops: [[N,] C, OW, KW]
void setPoolingNcwVectorSizes(
    linalg::LinalgOp op, int64_t vectorWidth,
    const CoralNPUTileSizeSelectionAnalysis &analysis,
    MutableArrayRef<int64_t> vectorParallelSizes,
    IREE::Codegen::DispatchLoweringPassPipeline &pipeline) {
  pipeline = IREE::Codegen::DispatchLoweringPassPipeline::CPUDoubleTilingExpert;

  const auto &loops = analysis.parallelLoops;

  int64_t owTile = vectorWidth;

  vectorParallelSizes[loops[0]] = owTile;
}

// tiling loops: [[N,] C, OH, OW, KH, KW]
void setPoolingNchwVectorSizes(
    linalg::LinalgOp op, int64_t vectorWidth,
    const CoralNPUTileSizeSelectionAnalysis &analysis,
    MutableArrayRef<int64_t> vectorParallelSizes,
    IREE::Codegen::DispatchLoweringPassPipeline &pipeline) {
  pipeline = IREE::Codegen::DispatchLoweringPassPipeline::CPUDoubleTilingExpert;

  const auto &loops = analysis.parallelLoops;

  int64_t owTile = vectorWidth;

  vectorParallelSizes[loops[0]] = owTile;
}

void setTransposeVectorSizes(
    linalg::LinalgOp op, int64_t vectorWidth,
    const CoralNPUTileSizeSelectionAnalysis &analysis,
    MutableArrayRef<int64_t> vectorParallelSizes,
    IREE::Codegen::DispatchLoweringPassPipeline &pipeline) {
  pipeline = IREE::Codegen::DispatchLoweringPassPipeline::CPUDoubleTilingExpert;

  const auto &loops = analysis.parallelLoops;

  int64_t tile = analysis.staticLoopRanges[loops[0]] % vectorWidth == 0
                     ? 8 * vectorWidth
                     : 6 * vectorWidth;

  vectorParallelSizes[loops[0]] = tile;
}

void setBroadcastVectorSizes(
    linalg::LinalgOp op, int64_t vectorWidth,
    const CoralNPUTileSizeSelectionAnalysis &analysis,
    MutableArrayRef<int64_t> vectorParallelSizes,
    IREE::Codegen::DispatchLoweringPassPipeline &pipeline) {
  pipeline = IREE::Codegen::DispatchLoweringPassPipeline::CPUDoubleTilingExpert;

  const auto &loops = analysis.parallelLoops;

  int64_t tile = analysis.staticLoopRanges[loops[0]] % vectorWidth == 0
                     ? 8 * vectorWidth
                     : 4 * vectorWidth;

  vectorParallelSizes[loops[0]] = tile;
}

void setReduceVectorSizes(
    linalg::LinalgOp op, int64_t vectorWidth,
    const CoralNPUTileSizeSelectionAnalysis &analysis,
    MutableArrayRef<int64_t> vectorParallelSizes,
    MutableArrayRef<int64_t> vectorReductionSizes,
    IREE::Codegen::DispatchLoweringPassPipeline &pipeline) {
  pipeline = IREE::Codegen::DispatchLoweringPassPipeline::CPUDoubleTilingExpert;

  if (!analysis.parallelLoops.empty()) {
    const auto &loops = analysis.parallelLoops;

    int64_t tile = 4 * vectorWidth;
    if (analysis.staticLoopRanges[loops[0]] % vectorWidth == 0) {
      tile = 8 * vectorWidth;
    }

    vectorParallelSizes[loops[0]] = tile;
    return;
  }

  if (!analysis.reductionLoops.empty()) {
    const auto &loops = analysis.reductionLoops;
    vectorReductionSizes[loops[0]] = 4 * vectorWidth;
  }
}

// tiling loops: [[B,] M, N, K]
void setMatmulVectorSizes(
    linalg::LinalgOp op, int64_t vectorWidth,
    const CoralNPUTileSizeSelectionAnalysis &analysis,
    MutableArrayRef<int64_t> vectorParallelSizes,
    IREE::Codegen::DispatchLoweringPassPipeline &pipeline) {
  pipeline = IREE::Codegen::DispatchLoweringPassPipeline::CPUDoubleTilingExpert;

  bool isMixedPrecision = false;
  if (op.getNumDpsInputs() >= 2 && op.getNumDpsInits() >= 1) {
    Type inType = op.getDpsInputOperand(0)->get().getType();
    Type outType = op.getDpsInitOperand(0)->get().getType();
    if (auto inShaped = dyn_cast<ShapedType>(inType)) {
      if (auto outShaped = dyn_cast<ShapedType>(outType)) {
        if (outShaped.getElementTypeBitWidth() >
            inShaped.getElementTypeBitWidth()) {
          isMixedPrecision = true;
        }
      }
    }
  }

  int64_t mTile = 4;
  int64_t nTile = std::min<int64_t>(16, 2 * vectorWidth);

  bool isBatch = analysis.parallelLoops.size() == 3 ||
                 isa<linalg::BatchMatmulOp, linalg::QuantizedBatchMatmulOp,
                     linalg::BatchReduceMatmulOp>(op);

  if (isa<linalg::QuantizedMatmulOp, linalg::QuantizedBatchMatmulOp>(op) ||
      isMixedPrecision) {
    nTile = 8;
  } else if (vectorWidth <= 4 && !isBatch) {
    mTile = 8;
    nTile = 2 * vectorWidth;
  }

  const auto &loops = analysis.parallelLoops;

  vectorParallelSizes[loops[0]] = nTile;
  vectorParallelSizes[loops[1]] = mTile;
}

// tiling loops: [[B,] M, K]
void setMatvecVectorSizes(
    linalg::LinalgOp op, int64_t vectorWidth,
    const CoralNPUTileSizeSelectionAnalysis &analysis,
    MutableArrayRef<int64_t> vectorParallelSizes,
    IREE::Codegen::DispatchLoweringPassPipeline &pipeline) {
  pipeline = IREE::Codegen::DispatchLoweringPassPipeline::CPUDoubleTilingExpert;

  int64_t mTile = 4 * vectorWidth;

  const auto &loops = analysis.parallelLoops;

  vectorParallelSizes[loops[0]] = mTile;
}

// tiling loops: [[B,] N, K]
void setVecmatVectorSizes(
    linalg::LinalgOp op, int64_t vectorWidth,
    const CoralNPUTileSizeSelectionAnalysis &analysis,
    MutableArrayRef<int64_t> vectorParallelSizes,
    IREE::Codegen::DispatchLoweringPassPipeline &pipeline) {
  pipeline = IREE::Codegen::DispatchLoweringPassPipeline::CPUDoubleTilingExpert;

  int64_t nTile = 4 * vectorWidth;

  const auto &loops = analysis.parallelLoops;

  vectorParallelSizes[loops[0]] = nTile;
}

void setDotVectorSizes(linalg::LinalgOp op, int64_t vectorWidth,
                       const CoralNPUTileSizeSelectionAnalysis &analysis,
                       MutableArrayRef<int64_t> vectorParallelSizes,
                       IREE::Codegen::DispatchLoweringPassPipeline &pipeline) {
  pipeline = IREE::Codegen::DispatchLoweringPassPipeline::CPUDoubleTilingExpert;
}

// tiling loops: [[B,] M1, N1, K1, M0, N0, K0]
void setMmt4DVectorSizes(
    linalg::LinalgOp op, int64_t vectorWidth,
    const CoralNPUTileSizeSelectionAnalysis &analysis,
    MutableArrayRef<int64_t> vectorParallelSizes,
    IREE::Codegen::DispatchLoweringPassPipeline &pipeline) {
  pipeline = IREE::Codegen::DispatchLoweringPassPipeline::CPUDoubleTilingExpert;
}

void setElementwiseArithBinaryVectorSizes(
    linalg::LinalgOp op, int64_t vectorWidth,
    const CoralNPUTileSizeSelectionAnalysis &analysis,
    MutableArrayRef<int64_t> vectorParallelSizes,
    IREE::Codegen::DispatchLoweringPassPipeline &pipeline) {
  pipeline = IREE::Codegen::DispatchLoweringPassPipeline::CPUDoubleTilingExpert;

  if (!analysis.parallelLoops.empty()) {
    const auto &loops = analysis.parallelLoops;
    int64_t tile = 4 * vectorWidth;
    vectorParallelSizes[loops[0]] = tile;
  }
}

void setElementwiseDivBinaryVectorSizes(
    linalg::LinalgOp op, int64_t vectorWidth,
    const CoralNPUTileSizeSelectionAnalysis &analysis,
    MutableArrayRef<int64_t> vectorParallelSizes,
    IREE::Codegen::DispatchLoweringPassPipeline &pipeline) {
  pipeline = IREE::Codegen::DispatchLoweringPassPipeline::CPUDoubleTilingExpert;

  if (!analysis.parallelLoops.empty()) {
    const auto &loops = analysis.parallelLoops;
    int64_t tile = 4 * vectorWidth;
    vectorParallelSizes[loops[0]] = tile;
  }
}

void setElementwiseUnaryArithVectorSizes(
    linalg::LinalgOp op, int64_t vectorWidth,
    const CoralNPUTileSizeSelectionAnalysis &analysis,
    MutableArrayRef<int64_t> vectorParallelSizes,
    IREE::Codegen::DispatchLoweringPassPipeline &pipeline) {
  pipeline = IREE::Codegen::DispatchLoweringPassPipeline::CPUDoubleTilingExpert;

  if (!analysis.parallelLoops.empty()) {
    const auto &loops = analysis.parallelLoops;
    int64_t tile = 4 * vectorWidth;
    vectorParallelSizes[loops[0]] = tile;
  }
}

void setElementwiseRoundingVectorSizes(
    linalg::LinalgOp op, int64_t vectorWidth,
    const CoralNPUTileSizeSelectionAnalysis &analysis,
    MutableArrayRef<int64_t> vectorParallelSizes,
    IREE::Codegen::DispatchLoweringPassPipeline &pipeline) {
  pipeline = IREE::Codegen::DispatchLoweringPassPipeline::CPUDoubleTilingExpert;

  if (!analysis.parallelLoops.empty()) {
    const auto &loops = analysis.parallelLoops;
    int64_t tile = 4 * vectorWidth;
    vectorParallelSizes[loops[0]] = tile;
  }
}

void setElementwiseRootVectorSizes(
    linalg::LinalgOp op, int64_t vectorWidth,
    const CoralNPUTileSizeSelectionAnalysis &analysis,
    MutableArrayRef<int64_t> vectorParallelSizes,
    IREE::Codegen::DispatchLoweringPassPipeline &pipeline) {
  pipeline = IREE::Codegen::DispatchLoweringPassPipeline::CPUDoubleTilingExpert;

  if (!analysis.parallelLoops.empty()) {
    const auto &loops = analysis.parallelLoops;
    int64_t tile = 4 * vectorWidth;
    vectorParallelSizes[loops[0]] = tile;
  }
}

void setElementwiseTranscendentalVectorSizes(
    linalg::LinalgOp op, int64_t vectorWidth,
    const CoralNPUTileSizeSelectionAnalysis &analysis,
    MutableArrayRef<int64_t> vectorParallelSizes,
    IREE::Codegen::DispatchLoweringPassPipeline &pipeline) {
  pipeline = IREE::Codegen::DispatchLoweringPassPipeline::CPUDoubleTilingExpert;

  if (!analysis.parallelLoops.empty()) {
    const auto &loops = analysis.parallelLoops;
    int64_t tile = vectorWidth;
    vectorParallelSizes[loops[0]] = tile;
  }
}

void setMapVectorSizes(linalg::LinalgOp op, int64_t vectorWidth,
                       const CoralNPUTileSizeSelectionAnalysis &analysis,
                       MutableArrayRef<int64_t> vectorParallelSizes,
                       IREE::Codegen::DispatchLoweringPassPipeline &pipeline) {
  pipeline = IREE::Codegen::DispatchLoweringPassPipeline::CPUDoubleTilingExpert;

  if (!analysis.parallelLoops.empty()) {
    const auto &loops = analysis.parallelLoops;
    int64_t tile = vectorWidth;
    vectorParallelSizes[loops[0]] = tile;
  }
}

void setGenericVectorSizes(
    linalg::GenericOp genericOp, int64_t vectorWidth,
    const CoralNPUTileSizeSelectionAnalysis &analysis,
    MutableArrayRef<int64_t> vectorParallelSizes,
    MutableArrayRef<int64_t> vectorReductionSizes,
    IREE::Codegen::DispatchLoweringPassPipeline &pipeline) {
  pipeline = IREE::Codegen::DispatchLoweringPassPipeline::CPUDoubleTilingExpert;

  if (!analysis.parallelLoops.empty()) {
    const auto &loops = analysis.parallelLoops;
    vectorParallelSizes[loops[0]] = vectorWidth;
    return;
  }

  if (!analysis.reductionLoops.empty()) {
    const auto &loops = analysis.reductionLoops;
    vectorReductionSizes[loops[0]] = 4 * vectorWidth;
  }
}

bool hasMultiplyAddBody(linalg::GenericOp genericOp) {
  Block *body = genericOp.getBody();
  if (!body) return false;
  bool hasMul = llvm::any_of(body->getOperations(), [](Operation &op) {
    return isa<arith::MulFOp, arith::MulIOp>(&op);
  });
  bool hasAdd = llvm::any_of(body->getOperations(), [](Operation &op) {
    return isa<arith::AddFOp, arith::AddIOp>(&op);
  });
  return hasMul && hasAdd;
}

bool isDepthwiseConv1DNcwCwGenericOp(linalg::GenericOp genericOp) {
  if (linalg::isaConvolutionOpOfType<linalg::DepthwiseConv1DNcwCwOp>(
          genericOp)) {
    return true;
  }
  if (genericOp.getNumDpsInputs() != 2 || genericOp.getNumDpsInits() != 1) {
    return false;
  }
  if (genericOp.getNumReductionLoops() != 1 ||
      genericOp.getNumParallelLoops() > 3) {
    return false;
  }
  if (!hasMultiplyAddBody(genericOp)) return false;

  AffineMap inMap =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInputOperand(0));
  AffineMap filterMap =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInputOperand(1));
  AffineMap outMap =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInitOperand(0));

  if (filterMap.getNumResults() != 2) return false;
  bool hasAdd = llvm::any_of(inMap.getResults(), [](AffineExpr expr) {
    return expr.getKind() == AffineExprKind::Add;
  });
  if (!hasAdd) return false;

  if (outMap.getNumResults() == 2 && filterMap.getNumResults() == 2) {
    return outMap.getResult(0) == filterMap.getResult(0);
  }
  return false;
}

bool isDepthwiseConv1DNwcWcGenericOp(linalg::GenericOp genericOp) {
  if (linalg::isaConvolutionOpOfType<linalg::DepthwiseConv1DNwcWcOp>(
          genericOp)) {
    return true;
  }
  if (genericOp.getNumDpsInputs() != 2 || genericOp.getNumDpsInits() != 1) {
    return false;
  }
  if (genericOp.getNumReductionLoops() != 1 ||
      genericOp.getNumParallelLoops() > 3) {
    return false;
  }
  if (!hasMultiplyAddBody(genericOp)) return false;

  AffineMap inMap =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInputOperand(0));
  AffineMap filterMap =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInputOperand(1));
  AffineMap outMap =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInitOperand(0));

  if (filterMap.getNumResults() != 2) return false;
  bool hasAdd = llvm::any_of(inMap.getResults(), [](AffineExpr expr) {
    return expr.getKind() == AffineExprKind::Add;
  });
  if (!hasAdd) return false;

  if (outMap.getNumResults() == 2 && filterMap.getNumResults() == 2) {
    return outMap.getResult(1) == filterMap.getResult(1);
  }
  return false;
}

bool isDepthwiseConv3DNdhwcDhwcmGenericOp(linalg::GenericOp genericOp) {
  if (linalg::isaConvolutionOpOfType<linalg::DepthwiseConv3DNdhwcDhwcmOp>(
          genericOp)) {
    return true;
  }
  if (genericOp.getNumDpsInputs() != 2 || genericOp.getNumDpsInits() != 1) {
    return false;
  }
  if (genericOp.getNumReductionLoops() != 3 ||
      genericOp.getNumParallelLoops() > 6) {
    return false;
  }
  if (!hasMultiplyAddBody(genericOp)) return false;

  AffineMap filterMap =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInputOperand(1));
  AffineMap outMap =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInitOperand(0));
  if (filterMap.getNumResults() != 5 || outMap.getNumResults() < 5)
    return false;

  return outMap.getResult(outMap.getNumResults() - 1) ==
             filterMap.getResult(4) &&
         outMap.getResult(outMap.getNumResults() - 2) == filterMap.getResult(3);
}

bool isDepthwiseConv3DNdhwcDhwcGenericOp(linalg::GenericOp genericOp) {
  if (linalg::isaConvolutionOpOfType<linalg::DepthwiseConv3DNdhwcDhwcOp>(
          genericOp)) {
    return true;
  }
  if (genericOp.getNumDpsInputs() != 2 || genericOp.getNumDpsInits() != 1) {
    return false;
  }
  if (genericOp.getNumReductionLoops() != 3 ||
      genericOp.getNumParallelLoops() > 5) {
    return false;
  }
  if (!hasMultiplyAddBody(genericOp)) return false;

  AffineMap inMap =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInputOperand(0));
  AffineMap filterMap =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInputOperand(1));
  AffineMap outMap =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInitOperand(0));

  if (filterMap.getNumResults() != 4 || outMap.getNumResults() < 4) {
    return false;
  }
  bool hasAdd = llvm::any_of(inMap.getResults(), [](AffineExpr expr) {
    return expr.getKind() == AffineExprKind::Add;
  });
  if (!hasAdd) return false;

  return outMap.getResult(outMap.getNumResults() - 1) == filterMap.getResult(3);
}

bool isDepthwiseConv2DNhwcHwcmGenericOp(linalg::GenericOp genericOp) {
  if (linalg::isaConvolutionOpOfType<linalg::DepthwiseConv2DNhwcHwcmOp>(
          genericOp) ||
      linalg::isaConvolutionOpOfType<linalg::DepthwiseConv2DNhwcHwcmQOp>(
          genericOp)) {
    return true;
  }
  if (genericOp.getNumDpsInputs() != 2 || genericOp.getNumDpsInits() != 1) {
    return false;
  }
  if (genericOp.getNumReductionLoops() != 2 ||
      genericOp.getNumParallelLoops() > 5) {
    return false;
  }
  if (!hasMultiplyAddBody(genericOp)) return false;

  AffineMap inMap =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInputOperand(0));
  AffineMap filterMap =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInputOperand(1));
  AffineMap outMap =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInitOperand(0));

  if (filterMap.getNumResults() != 4 || outMap.getNumResults() < 4) {
    return false;
  }
  bool hasAdd = llvm::any_of(inMap.getResults(), [](AffineExpr expr) {
    return expr.getKind() == AffineExprKind::Add;
  });
  if (!hasAdd) return false;

  return outMap.getResult(outMap.getNumResults() - 1) ==
             filterMap.getResult(3) &&
         outMap.getResult(outMap.getNumResults() - 2) == filterMap.getResult(2);
}

bool isDepthwiseConv2DNhwcHwcGenericOp(linalg::GenericOp genericOp) {
  if (linalg::isaConvolutionOpOfType<linalg::DepthwiseConv2DNhwcHwcOp>(
          genericOp)) {
    return true;
  }
  if (genericOp.getNumDpsInputs() != 2 || genericOp.getNumDpsInits() != 1) {
    return false;
  }
  if (genericOp.getNumReductionLoops() != 2 ||
      genericOp.getNumParallelLoops() > 4) {
    return false;
  }
  if (!hasMultiplyAddBody(genericOp)) return false;

  AffineMap inMap =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInputOperand(0));
  AffineMap filterMap =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInputOperand(1));
  AffineMap outMap =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInitOperand(0));

  if (filterMap.getNumResults() != 3 || outMap.getNumResults() < 3) {
    return false;
  }
  bool hasAdd = llvm::any_of(inMap.getResults(), [](AffineExpr expr) {
    return expr.getKind() == AffineExprKind::Add;
  });
  if (!hasAdd) return false;

  return outMap.getResult(outMap.getNumResults() - 1) == filterMap.getResult(2);
}

bool isaPoolingNdhwcGenericOp(linalg::GenericOp genericOp) {
  if (linalg::isaConvolutionOpOfType<linalg::PoolingNdhwcSumOp>(genericOp) ||
      linalg::isaConvolutionOpOfType<linalg::PoolingNdhwcMaxOp>(genericOp) ||
      linalg::isaConvolutionOpOfType<linalg::PoolingNdhwcMinOp>(genericOp)) {
    return true;
  }
  if (genericOp.getNumDpsInputs() != 2 || genericOp.getNumDpsInits() != 1) {
    return false;
  }
  if (genericOp.getNumReductionLoops() != 3 ||
      genericOp.getNumParallelLoops() > 5) {
    return false;
  }

  AffineMap filterMap =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInputOperand(1));
  AffineMap outMap =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInitOperand(0));
  return filterMap.getNumResults() == 3 && outMap.getNumResults() >= 4;
}

bool isaTransposeGenericOp(linalg::GenericOp genericOp) {
  if (genericOp.getNumDpsInputs() != 1 || genericOp.getNumDpsInits() != 1) {
    return false;
  }
  if (genericOp.getNumParallelLoops() != genericOp.getNumLoops()) {
    return false;
  }

  AffineMap inMap =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInputOperand(0));
  AffineMap outMap =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInitOperand(0));
  if (!inMap.isPermutation() || !outMap.isPermutation() || inMap == outMap) {
    return false;
  }

  Block *body = genericOp.getBody();
  if (!body || body->empty()) return false;
  auto yieldOp = dyn_cast<linalg::YieldOp>(body->getTerminator());
  return yieldOp && yieldOp.getNumOperands() == 1 &&
         yieldOp.getOperand(0) == body->getArgument(0);
}

bool isaBroadcastGenericOp(linalg::GenericOp genericOp) {
  if (genericOp.getNumDpsInputs() != 1 || genericOp.getNumDpsInits() != 1) {
    return false;
  }
  if (genericOp.getNumParallelLoops() != genericOp.getNumLoops()) {
    return false;
  }

  AffineMap inMap =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInputOperand(0));
  AffineMap outMap =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInitOperand(0));
  if (inMap.getNumResults() >= outMap.getNumResults()) {
    return false;
  }
  if (!inMap.isProjectedPermutation() || !outMap.isPermutation()) {
    return false;
  }

  Block *body = genericOp.getBody();
  if (!body || body->empty()) return false;
  auto yieldOp = dyn_cast<linalg::YieldOp>(body->getTerminator());
  return yieldOp && yieldOp.getNumOperands() == 1 &&
         yieldOp.getOperand(0) == body->getArgument(0);
}

bool isaReduceGenericOp(linalg::GenericOp genericOp) {
  if (genericOp.getNumDpsInputs() != 1 || genericOp.getNumDpsInits() != 1) {
    return false;
  }
  if (genericOp.getNumReductionLoops() < 1) {
    return false;
  }

  AffineMap inMap =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInputOperand(0));
  AffineMap outMap =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInitOperand(0));
  if (outMap.getNumResults() >= inMap.getNumResults()) {
    return false;
  }
  return (inMap.isPermutation() || inMap.isProjectedPermutation()) &&
         (outMap.isPermutation() || outMap.isProjectedPermutation());
}

bool isMatmulGenericOp(linalg::GenericOp genericOp) {
  if (linalg::isaContractionOpInterface(genericOp)) {
    return true;
  }
  if (genericOp.getNumDpsInputs() < 2 || genericOp.getNumDpsInits() != 1) {
    return false;
  }
  if (genericOp.getNumParallelLoops() != 2 ||
      genericOp.getNumReductionLoops() != 1) {
    return false;
  }
  if (!hasMultiplyAddBody(genericOp)) return false;

  AffineMap in0Map =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInputOperand(0));
  AffineMap in1Map =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInputOperand(1));
  AffineMap outMap =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInitOperand(0));

  if (outMap.getNumResults() == 2 && in0Map.getNumResults() >= 2 &&
      in1Map.getNumResults() >= 2) {
    return outMap.getResult(0) == in0Map.getResult(0) &&
           outMap.getResult(1) == in1Map.getResult(1);
  }
  return false;
}

bool isBatchMatmulGenericOp(linalg::GenericOp genericOp) {
  if (genericOp.getNumDpsInputs() < 2 || genericOp.getNumDpsInits() != 1) {
    return false;
  }
  if (genericOp.getNumParallelLoops() != 3 ||
      genericOp.getNumReductionLoops() != 1) {
    return false;
  }
  if (!hasMultiplyAddBody(genericOp)) return false;

  AffineMap in0Map =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInputOperand(0));
  AffineMap in1Map =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInputOperand(1));
  AffineMap outMap =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInitOperand(0));

  if (outMap.getNumResults() == 3 && in0Map.getNumResults() >= 3 &&
      in1Map.getNumResults() >= 3) {
    return outMap.getResult(0) == in0Map.getResult(0) &&
           outMap.getResult(0) == in1Map.getResult(0) &&
           outMap.getResult(1) == in0Map.getResult(1) &&
           outMap.getResult(2) == in1Map.getResult(2);
  }
  return false;
}

bool isaElementwiseDivGenericOp(linalg::GenericOp genericOp) {
  if (genericOp.getNumDpsInputs() != 2 || genericOp.getNumDpsInits() != 1) {
    return false;
  }
  if (genericOp.getNumParallelLoops() != genericOp.getNumLoops()) {
    return false;
  }
  if (!linalg::isElementwise(genericOp)) {
    return false;
  }

  Block *body = genericOp.getBody();
  if (!body) return false;
  return llvm::any_of(body->getOperations(), [](Operation &op) {
    return isa<arith::DivFOp, arith::DivSIOp, arith::DivUIOp>(&op);
  });
}

bool isaElementwiseArithBinaryGenericOp(linalg::GenericOp genericOp) {
  if (genericOp.getNumDpsInputs() != 2 || genericOp.getNumDpsInits() != 1) {
    return false;
  }
  if (genericOp.getNumParallelLoops() != genericOp.getNumLoops()) {
    return false;
  }
  if (!linalg::isElementwise(genericOp)) {
    return false;
  }

  Block *body = genericOp.getBody();
  if (!body) return false;
  return llvm::any_of(body->getOperations(), [](Operation &op) {
    return isa<arith::AddFOp, arith::AddIOp, arith::SubFOp, arith::SubIOp,
               arith::MulFOp, arith::MulIOp, arith::MinimumFOp,
               arith::MinNumFOp, arith::MinSIOp, arith::MinUIOp,
               arith::MaximumFOp, arith::MaxNumFOp, arith::MaxSIOp,
               arith::MaxUIOp>(&op);
  });
}

bool isaElementwiseUnaryArithGenericOp(linalg::GenericOp genericOp) {
  if (genericOp.getNumDpsInputs() != 1 || genericOp.getNumDpsInits() != 1) {
    return false;
  }
  if (genericOp.getNumParallelLoops() != genericOp.getNumLoops()) {
    return false;
  }
  if (!linalg::isElementwise(genericOp)) {
    return false;
  }

  Block *body = genericOp.getBody();
  if (!body) return false;
  return llvm::any_of(body->getOperations(), [](Operation &op) {
    return isa<math::AbsFOp, math::AbsIOp, arith::NegFOp>(&op);
  });
}

bool isaElementwiseRoundingGenericOp(linalg::GenericOp genericOp) {
  if (genericOp.getNumDpsInputs() != 1 || genericOp.getNumDpsInits() != 1) {
    return false;
  }
  if (genericOp.getNumParallelLoops() != genericOp.getNumLoops()) {
    return false;
  }
  if (!linalg::isElementwise(genericOp)) {
    return false;
  }

  Block *body = genericOp.getBody();
  if (!body) return false;
  return llvm::any_of(body->getOperations(), [](Operation &op) {
    return isa<math::CeilOp, math::FloorOp, math::RoundOp, math::RoundEvenOp>(
        &op);
  });
}

bool isaElementwiseRootGenericOp(linalg::GenericOp genericOp) {
  if (genericOp.getNumDpsInputs() != 1 || genericOp.getNumDpsInits() != 1) {
    return false;
  }
  if (genericOp.getNumParallelLoops() != genericOp.getNumLoops()) {
    return false;
  }
  if (!linalg::isElementwise(genericOp)) {
    return false;
  }

  Block *body = genericOp.getBody();
  if (!body) return false;
  return llvm::any_of(body->getOperations(), [](Operation &op) {
    return isa<math::SqrtOp, math::RsqrtOp>(&op);
  });
}

bool isaElementwiseTranscendentalGenericOp(linalg::GenericOp genericOp) {
  if (genericOp.getNumDpsInputs() < 1 || genericOp.getNumDpsInits() != 1) {
    return false;
  }
  if (genericOp.getNumParallelLoops() != genericOp.getNumLoops()) {
    return false;
  }
  if (!linalg::isElementwise(genericOp)) {
    return false;
  }

  Block *body = genericOp.getBody();
  if (!body) return false;
  return llvm::any_of(body->getOperations(), [](Operation &op) {
    return isa<math::ExpOp, math::Exp2Op, math::LogOp, math::Log2Op,
               math::Log10Op, math::Log1pOp, math::TanhOp, math::ErfOp,
               math::PowFOp, math::SinOp, math::CosOp, math::TanOp,
               math::AsinOp, math::AcosOp, math::AtanOp, math::Atan2Op,
               math::SinhOp, math::CoshOp>(&op);
  });
}

bool isaMapGenericOp(linalg::GenericOp genericOp) {
  if (genericOp.getNumParallelLoops() != genericOp.getNumLoops()) {
    return false;
  }
  if (!linalg::isElementwise(genericOp)) {
    return false;
  }
  return llvm::all_of(genericOp.getIndexingMapsArray(),
                      [](AffineMap map) { return map.isIdentity(); });
}

void setLinalgGenericVectorSizes(
    linalg::GenericOp genericOp, int64_t vectorWidth,
    const CoralNPUTileSizeSelectionAnalysis &analysis,
    MutableArrayRef<int64_t> vectorParallelSizes,
    MutableArrayRef<int64_t> vectorReductionSizes,
    IREE::Codegen::DispatchLoweringPassPipeline &pipeline) {
  if (isaTransposeGenericOp(genericOp)) {
    return setTransposeVectorSizes(genericOp, vectorWidth, analysis,
                                   vectorParallelSizes, pipeline);
  }

  if (isaBroadcastGenericOp(genericOp)) {
    return setBroadcastVectorSizes(genericOp, vectorWidth, analysis,
                                   vectorParallelSizes, pipeline);
  }

  if (isaReduceGenericOp(genericOp)) {
    return setReduceVectorSizes(genericOp, vectorWidth, analysis,
                                vectorParallelSizes, vectorReductionSizes,
                                pipeline);
  }

  if (isDepthwiseConv1DNwcWcGenericOp(genericOp)) {
    return setDepthwiseConv1DNwcWcVectorSizes(genericOp, vectorWidth, analysis,
                                              vectorParallelSizes, pipeline);
  }

  if (isDepthwiseConv1DNcwCwGenericOp(genericOp)) {
    return setDepthwiseConv1DNcwCwVectorSizes(genericOp, vectorWidth, analysis,
                                              vectorParallelSizes, pipeline);
  }

  if (isDepthwiseConv3DNdhwcDhwcmGenericOp(genericOp)) {
    return setDepthwiseConv3DNdhwcDhwcmVectorSizes(
        genericOp, vectorWidth, analysis, vectorParallelSizes, pipeline);
  }

  if (isDepthwiseConv3DNdhwcDhwcGenericOp(genericOp)) {
    return setDepthwiseConv3DNdhwcDhwcVectorSizes(
        genericOp, vectorWidth, analysis, vectorParallelSizes, pipeline);
  }

  if (isDepthwiseConv2DNhwcHwcmGenericOp(genericOp)) {
    return setDepthwiseConv2DNhwcHwcmVectorSizes(
        genericOp, vectorWidth, analysis, vectorParallelSizes, pipeline);
  }

  if (isDepthwiseConv2DNhwcHwcGenericOp(genericOp)) {
    return setDepthwiseConv2DNhwcHwcVectorSizes(
        genericOp, vectorWidth, analysis, vectorParallelSizes, pipeline);
  }

  if (isaPoolingNdhwcGenericOp(genericOp)) {
    return setPoolingNdhwcVectorSizes(genericOp, vectorWidth, analysis,
                                      vectorParallelSizes, pipeline);
  }

  if (isaElementwiseDivGenericOp(genericOp)) {
    return setElementwiseDivBinaryVectorSizes(genericOp, vectorWidth, analysis,
                                              vectorParallelSizes, pipeline);
  }

  if (isaElementwiseArithBinaryGenericOp(genericOp)) {
    return setElementwiseArithBinaryVectorSizes(
        genericOp, vectorWidth, analysis, vectorParallelSizes, pipeline);
  }

  if (isaElementwiseUnaryArithGenericOp(genericOp)) {
    return setElementwiseUnaryArithVectorSizes(genericOp, vectorWidth, analysis,
                                               vectorParallelSizes, pipeline);
  }

  if (isaElementwiseRoundingGenericOp(genericOp)) {
    return setElementwiseRoundingVectorSizes(genericOp, vectorWidth, analysis,
                                             vectorParallelSizes, pipeline);
  }

  if (isaElementwiseRootGenericOp(genericOp)) {
    return setElementwiseRootVectorSizes(genericOp, vectorWidth, analysis,
                                         vectorParallelSizes, pipeline);
  }
  if (isaElementwiseTranscendentalGenericOp(genericOp)) {
    return setElementwiseTranscendentalVectorSizes(
        genericOp, vectorWidth, analysis, vectorParallelSizes, pipeline);
  }

  if (isaMapGenericOp(genericOp)) {
    return setMapVectorSizes(genericOp, vectorWidth, analysis,
                             vectorParallelSizes, pipeline);
  }

  if (isMatmulGenericOp(genericOp) || isBatchMatmulGenericOp(genericOp)) {
    return setMatmulVectorSizes(genericOp, vectorWidth, analysis,
                                vectorParallelSizes, pipeline);
  }

  return setGenericVectorSizes(genericOp, vectorWidth, analysis,
                               vectorParallelSizes, vectorReductionSizes,
                               pipeline);
}

void setLinalgOpVectorSizes(
    linalg::LinalgOp linalgOp, int64_t vectorWidth,
    const CoralNPUTileSizeSelectionAnalysis &analysis,
    MutableArrayRef<int64_t> vectorParallelSizes,
    MutableArrayRef<int64_t> vectorReductionSizes,
    IREE::Codegen::DispatchLoweringPassPipeline &pipeline) {
  if (isa<linalg::Conv2DNhwcHwcfOp, linalg::Conv2DNhwcHwcfQOp>(linalgOp)) {
    return setConv2DNhwcHwcfVectorSizes(linalgOp, vectorWidth, analysis,
                                        vectorParallelSizes, pipeline);
  }

  if (isa<linalg::Conv2DNchwFchwOp, linalg::Conv2DNchwFchwQOp>(linalgOp)) {
    return setConv2DNchwFchwVectorSizes(linalgOp, vectorWidth, analysis,
                                        vectorParallelSizes, pipeline);
  }

  if (isa<linalg::Conv2DNhwcFhwcOp, linalg::Conv2DNhwcFhwcQOp>(linalgOp)) {
    return setConv2DNhwcFhwcVectorSizes(linalgOp, vectorWidth, analysis,
                                        vectorParallelSizes, pipeline);
  }

  if (isa<linalg::Conv1DNwcWcfOp>(linalgOp)) {
    return setConv1DNwcWcfVectorSizes(linalgOp, vectorWidth, analysis,
                                      vectorParallelSizes, pipeline);
  }

  if (isa<linalg::Conv1DNcwFcwOp>(linalgOp)) {
    return setConv1DNcwFcwVectorSizes(linalgOp, vectorWidth, analysis,
                                      vectorParallelSizes, pipeline);
  }

  if (isa<linalg::Conv1DOp>(linalgOp)) {
    return setConv1DVectorSizes(linalgOp, vectorWidth, analysis,
                                vectorParallelSizes, pipeline);
  }

  if (isa<linalg::Conv2DOp>(linalgOp)) {
    return setConv2DVectorSizes(linalgOp, vectorWidth, analysis,
                                vectorParallelSizes, pipeline);
  }

  if (isa<linalg::Conv3DOp>(linalgOp)) {
    return setConv3DVectorSizes(linalgOp, vectorWidth, analysis,
                                vectorParallelSizes, pipeline);
  }

  if (isa<linalg::Conv3DNdhwcDhwcfOp, linalg::Conv3DNdhwcDhwcfQOp>(linalgOp)) {
    return setConv3DNdhwcDhwcfVectorSizes(linalgOp, vectorWidth, analysis,
                                          vectorParallelSizes, pipeline);
  }

  if (isa<linalg::Conv3DNcdhwFcdhwOp>(linalgOp)) {
    return setConv3DNcdhwFcdhwVectorSizes(linalgOp, vectorWidth, analysis,
                                          vectorParallelSizes, pipeline);
  }

  if (isa<linalg::DepthwiseConv2DNhwcHwcOp>(linalgOp)) {
    return setDepthwiseConv2DNhwcHwcVectorSizes(linalgOp, vectorWidth, analysis,
                                                vectorParallelSizes, pipeline);
  }

  if (isa<linalg::DepthwiseConv1DNwcWcOp>(linalgOp)) {
    return setDepthwiseConv1DNwcWcVectorSizes(linalgOp, vectorWidth, analysis,
                                              vectorParallelSizes, pipeline);
  }

  if (isa<linalg::DepthwiseConv1DNcwCwOp>(linalgOp)) {
    return setDepthwiseConv1DNcwCwVectorSizes(linalgOp, vectorWidth, analysis,
                                              vectorParallelSizes, pipeline);
  }

  if (isa<linalg::DepthwiseConv1DNwcWcmOp>(linalgOp) ||
      linalg::isaConvolutionOpOfType<linalg::DepthwiseConv1DNwcWcmOp>(
          linalgOp)) {
    return setDepthwiseConv1DNwcWcmVectorSizes(linalgOp, vectorWidth, analysis,
                                               vectorParallelSizes, pipeline);
  }

  if (isa<linalg::PoolingNhwcSumOp, linalg::PoolingNhwcMaxOp,
          linalg::PoolingNhwcMinOp, linalg::PoolingNhwcMaxUnsignedOp,
          linalg::PoolingNhwcMinUnsignedOp>(linalgOp)) {
    return setPoolingNhwcVectorSizes(linalgOp, vectorWidth, analysis,
                                     vectorParallelSizes, pipeline);
  }

  if (isa<linalg::DepthwiseConv3DNdhwcDhwcOp>(linalgOp)) {
    return setDepthwiseConv3DNdhwcDhwcVectorSizes(
        linalgOp, vectorWidth, analysis, vectorParallelSizes, pipeline);
  }

  if (isa<linalg::DepthwiseConv3DNdhwcDhwcmOp>(linalgOp)) {
    return setDepthwiseConv3DNdhwcDhwcmVectorSizes(
        linalgOp, vectorWidth, analysis, vectorParallelSizes, pipeline);
  }

  if (isa<linalg::Conv2DNhwgcGfhwcOp, linalg::Conv2DNhwgcGfhwcQOp>(linalgOp)) {
    return setConv2DNhwgcGfhwcVectorSizes(linalgOp, vectorWidth, analysis,
                                          vectorParallelSizes, pipeline);
  }

  if (isa<linalg::Conv2DNgchwGfchwOp, linalg::Conv2DNgchwGfchwQOp,
          linalg::Conv2DNgchwFgchwOp>(linalgOp)) {
    return setConv2DNgchwGfchwVectorSizes(linalgOp, vectorWidth, analysis,
                                          vectorParallelSizes, pipeline);
  }

  if (isa<linalg::DepthwiseConv2DNhwcHwcmOp,
          linalg::DepthwiseConv2DNhwcHwcmQOp>(linalgOp)) {
    return setDepthwiseConv2DNhwcHwcmVectorSizes(
        linalgOp, vectorWidth, analysis, vectorParallelSizes, pipeline);
  }

  if (isa<linalg::PoolingNwcSumOp, linalg::PoolingNwcMaxOp,
          linalg::PoolingNwcMaxUnsignedOp, linalg::PoolingNwcMinOp,
          linalg::PoolingNwcMinUnsignedOp>(linalgOp) ||
      linalg::isaConvolutionOpOfType<linalg::PoolingNwcSumOp>(linalgOp) ||
      linalg::isaConvolutionOpOfType<linalg::PoolingNwcMaxOp>(linalgOp) ||
      linalg::isaConvolutionOpOfType<linalg::PoolingNwcMaxUnsignedOp>(
          linalgOp) ||
      linalg::isaConvolutionOpOfType<linalg::PoolingNwcMinOp>(linalgOp) ||
      linalg::isaConvolutionOpOfType<linalg::PoolingNwcMinUnsignedOp>(
          linalgOp)) {
    return setPoolingNwcVectorSizes(linalgOp, vectorWidth, analysis,
                                    vectorParallelSizes, pipeline);
  }

  if (isa<linalg::PoolingNcwSumOp, linalg::PoolingNcwMaxOp>(linalgOp) ||
      linalg::isaConvolutionOpOfType<linalg::PoolingNcwSumOp>(linalgOp) ||
      linalg::isaConvolutionOpOfType<linalg::PoolingNcwMaxOp>(linalgOp)) {
    return setPoolingNcwVectorSizes(linalgOp, vectorWidth, analysis,
                                    vectorParallelSizes, pipeline);
  }

  if (isa<linalg::PoolingNchwSumOp, linalg::PoolingNchwMaxOp>(linalgOp) ||
      linalg::isaConvolutionOpOfType<linalg::PoolingNchwSumOp>(linalgOp) ||
      linalg::isaConvolutionOpOfType<linalg::PoolingNchwMaxOp>(linalgOp)) {
    return setPoolingNchwVectorSizes(linalgOp, vectorWidth, analysis,
                                     vectorParallelSizes, pipeline);
  }

  if (isa<linalg::PoolingNdhwcSumOp, linalg::PoolingNdhwcMaxOp,
          linalg::PoolingNdhwcMinOp>(linalgOp)) {
    return setPoolingNdhwcVectorSizes(linalgOp, vectorWidth, analysis,
                                      vectorParallelSizes, pipeline);
  }

  if (isa<linalg::MatmulOp, linalg::QuantizedMatmulOp, linalg::ContractOp,
          linalg::BatchMatmulOp, linalg::QuantizedBatchMatmulOp,
          linalg::BatchReduceMatmulOp>(linalgOp)) {
    return setMatmulVectorSizes(linalgOp, vectorWidth, analysis,
                                vectorParallelSizes, pipeline);
  }

  if (isa<linalg::MatvecOp>(linalgOp)) {
    return setMatvecVectorSizes(linalgOp, vectorWidth, analysis,
                                vectorParallelSizes, pipeline);
  }

  if (isa<linalg::BatchMatvecOp>(linalgOp)) {
    return setMatvecVectorSizes(linalgOp, vectorWidth, analysis,
                                vectorParallelSizes, pipeline);
  }

  if (isa<linalg::VecmatOp>(linalgOp)) {
    return setVecmatVectorSizes(linalgOp, vectorWidth, analysis,
                                vectorParallelSizes, pipeline);
  }

  if (isa<linalg::BatchVecmatOp>(linalgOp)) {
    return setVecmatVectorSizes(linalgOp, vectorWidth, analysis,
                                vectorParallelSizes, pipeline);
  }

  if (isa<linalg::DotOp>(linalgOp)) {
    return setDotVectorSizes(linalgOp, vectorWidth, analysis,
                             vectorParallelSizes, pipeline);
  }

  if (isa<linalg::Mmt4DOp>(linalgOp)) {
    return setMmt4DVectorSizes(linalgOp, vectorWidth, analysis,
                               vectorParallelSizes, pipeline);
  }

  if (isa<linalg::BatchMmt4DOp>(linalgOp)) {
    return setMmt4DVectorSizes(linalgOp, vectorWidth, analysis,
                               vectorParallelSizes, pipeline);
  }

  if (isa<linalg::AddOp, linalg::SubOp, linalg::MulOp, linalg::MinOp,
          linalg::MaxOp, linalg::ElementwiseOp>(linalgOp)) {
    return setElementwiseArithBinaryVectorSizes(linalgOp, vectorWidth, analysis,
                                                vectorParallelSizes, pipeline);
  }

  if (isa<linalg::DivOp, linalg::DivUnsignedOp>(linalgOp)) {
    return setElementwiseDivBinaryVectorSizes(linalgOp, vectorWidth, analysis,
                                              vectorParallelSizes, pipeline);
  }

  if (isa<linalg::AbsOp, linalg::NegFOp, linalg::SquareOp, linalg::CopyOp>(
          linalgOp)) {
    return setElementwiseUnaryArithVectorSizes(linalgOp, vectorWidth, analysis,
                                               vectorParallelSizes, pipeline);
  }

  if (isa<linalg::CeilOp, linalg::FloorOp, linalg::RoundOp>(linalgOp)) {
    return setElementwiseRoundingVectorSizes(linalgOp, vectorWidth, analysis,
                                             vectorParallelSizes, pipeline);
  }

  if (isa<linalg::SqrtOp, linalg::RsqrtOp, linalg::ReciprocalOp>(linalgOp)) {
    return setElementwiseRootVectorSizes(linalgOp, vectorWidth, analysis,
                                         vectorParallelSizes, pipeline);
  }

  if (isa<linalg::ExpOp, linalg::LogOp, linalg::TanhOp, linalg::ErfOp,
          linalg::PowFOp>(linalgOp)) {
    return setElementwiseTranscendentalVectorSizes(
        linalgOp, vectorWidth, analysis, vectorParallelSizes, pipeline);
  }

  if (isa<linalg::MapOp>(linalgOp)) {
    return setMapVectorSizes(linalgOp, vectorWidth, analysis,
                             vectorParallelSizes, pipeline);
  }

  if (isa<linalg::TransposeOp>(linalgOp)) {
    return setTransposeVectorSizes(linalgOp, vectorWidth, analysis,
                                   vectorParallelSizes, pipeline);
  }

  if (isa<linalg::BroadcastOp>(linalgOp)) {
    return setBroadcastVectorSizes(linalgOp, vectorWidth, analysis,
                                   vectorParallelSizes, pipeline);
  }

  if (isa<linalg::ReduceOp>(linalgOp)) {
    return setReduceVectorSizes(linalgOp, vectorWidth, analysis,
                                vectorParallelSizes, vectorReductionSizes,
                                pipeline);
  }
}

// Set vectorParallelSizes and vectorReductionSizes elements to 0 (indicating
// they should not be tiled) if their calculated tile size is greater or equal
// their original size.
void capVectorSizes(const CoralNPUTileSizeSelectionAnalysis &analysis,
                    MutableArrayRef<int64_t> vectorParallelSizes,
                    MutableArrayRef<int64_t> vectorReductionSizes) {
  for (auto [tile, range] :
       llvm::zip_equal(vectorParallelSizes, analysis.staticLoopRanges)) {
    if (range != 0 && tile >= range) tile = 0;  // do not tile
  }

  for (auto [tile, range] :
       llvm::zip_equal(vectorReductionSizes, analysis.staticLoopRanges)) {
    if (range != 0 && tile >= range) tile = 0;  // do not tile
  }
}

void setVectorSizes(TilingInterface tilingOp, int64_t vectorWidth,
                    const CoralNPUTileSizeSelectionAnalysis &analysis,
                    MutableArrayRef<int64_t> vectorParallelSizes,
                    MutableArrayRef<int64_t> vectorReductionSizes,
                    IREE::Codegen::DispatchLoweringPassPipeline &pipeline) {
  // set very strict default/fallback values
  pipeline = IREE::Codegen::DispatchLoweringPassPipeline::CPUDoubleTilingExpert;

  for (size_t vecIdx : analysis.parallelLoops) {
    vectorParallelSizes[vecIdx] = 1;
  }

  for (size_t vecIdx : analysis.reductionLoops) {
    vectorReductionSizes[vecIdx] = 1;
  }

  // try to specialize the values
  if (auto genericOp = dyn_cast<linalg::GenericOp>(tilingOp.getOperation())) {
    setLinalgGenericVectorSizes(genericOp, vectorWidth, analysis,
                                vectorParallelSizes, vectorReductionSizes,
                                pipeline);
  } else if (auto linalgOp =
                 dyn_cast<linalg::LinalgOp>(tilingOp.getOperation())) {
    setLinalgOpVectorSizes(linalgOp, vectorWidth, analysis, vectorParallelSizes,
                           vectorReductionSizes, pipeline);
  }

  capVectorSizes(analysis, vectorParallelSizes, vectorReductionSizes);
}

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

    auto &analysis = getAnalysis<CoralNPUTileSizeSelectionAnalysis>();
    if (failed(analysis.status)) {
      signalPassFailure();
      return;
    }

    auto tilingOp = analysis.rootTilingOp;

    // Resolve VLEN and compute vector width
    int64_t vlenBits = vectorRegisterWidthBits;
    if (vlenBits == 0) {
      vlenBits = getVlenFromTargetFeatures(funcOp).value_or(128);
    }
    int64_t vectorWidth = (vlenBits / 8) / analysis.elemSizeBytes;
    assert(vectorWidth > 0);

    int64_t numLoops = analysis.staticLoopRanges.size();
    SmallVector<int64_t> vectorParallelSizes(numLoops, 0);
    SmallVector<int64_t> vectorReductionSizes(numLoops, 0);

    IREE::Codegen::DispatchLoweringPassPipeline pipeline =
        IREE::Codegen::DispatchLoweringPassPipeline::None;

    setVectorSizes(tilingOp, vectorWidth, analysis, vectorParallelSizes,
                   vectorReductionSizes, pipeline);

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

    auto translationInfo =
        IREE::Codegen::TranslationInfoAttr::get(context, pipeline);

    auto compilationInfo = IREE::Codegen::CompilationInfoAttr::get(
        context, loweringConfig, translationInfo);

    setCompilationInfo(tilingOp, compilationInfo);

    markAnalysesPreserved<CoralNPUTileSizeSelectionAnalysis>();
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

}  // namespace mlir::coralnpu_compiler
