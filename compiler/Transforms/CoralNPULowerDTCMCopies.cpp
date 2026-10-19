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
// CoralNPU DTCM copies: lowers the copies between memory and DTCM
// (#iree_codegen.workgroup_local memrefs) to loops of vector loads and stores.
//
// The pass runs after the codegen pipeline (bufferization and the vector
// lowering), in the pre-LLVM hook of the CoralNPU target. There, the copies of
// CoralNPUPromoteOperands are linalg.generic copies of whole tiles: the
// materialize_in_destination ops of the input tiles, and the stores of the
// staged results. Without this pass, the pre-LLVM hook makes a copy of fewer
// than 1024 elements one vector (GenericVectorizationPass), and IREE lowers the
// other copies to scalar loops (MemrefCopyToLinalgPass and
// ConvertLinalgToLoopsPass).
//
// TODO(DMA): this pass is the place where the copies between memory and DTCM
// must become DMA transfers. Each copy here moves one tile (the rows of a slice
// of a binding) between memory and DTCM. With DMA:
// - Make one 1D DMA descriptor for each row (the DMA is 1D). Descriptors are
//   32-byte aligned, beats are at most 16 bytes, and the length must be a
//   multiple of the beat.
// - Start the transfer of an input tile before the computation, and wait for
//   it (coralnpu_dma_wait_done) before the first read of the DTCM buffer. With
//   double buffering, start the transfer of the next tile (e.g. the next K
//   chunk) while the current tile computes.
// - Start the transfer of a staged result after its last write, and wait for
//   it before the next write to the DTCM buffer (or at the end of the
//   dispatch).
// - Keep the vector loops of this pass as the fallback for the copies that the
//   DMA cannot do (e.g. bad alignment or length).
//===----------------------------------------------------------------------===//

#include "compiler/Transforms/CoralNPUTileSizeSelectionUtils.h"
#include "compiler/Transforms/Passes.h"

// IREE:
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"

// MLIR:
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

// LLVM:
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"

// Standard library:
#include <algorithm>
#include <memory>
#include <optional>

using namespace mlir;
using namespace mlir::iree_compiler;

namespace mlir::coralnpu_compiler {

#define GEN_PASS_DEF_CORALNPULOWERDTCMCOPIES
#include "compiler/Transforms/Passes.h.inc"

namespace {

// A copy of the memref `source` to the memref `target` (same shape) by `op`.
struct MemRefCopy {
  Operation *op;
  Value source;
  Value target;
};

// Returns the copy that `op` does, if `op` only copies a memref to another
// memref: a memref.copy, or a linalg op on buffers that is a copy (e.g. the
// linalg.generic copies that bufferization makes).
std::optional<MemRefCopy> matchCopy(Operation *op) {
  // A memref.copy is a copy.
  if (auto copy = dyn_cast<memref::CopyOp>(op)) {
    return MemRefCopy{op, copy.getSource(), copy.getTarget()};
  }
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  // A linalg copy has parallel loops, identity maps, and a body that only
  // yields the input element.
  if (!linalgOp || !linalgOp.hasPureBufferSemantics() ||
      !linalg::isaCopyOpInterface(linalgOp)) {
    return std::nullopt;
  }
  return MemRefCopy{op, linalgOp.getDpsInputOperand(0)->get(),
                    linalgOp.getDpsInitOperand(0)->get()};
}

// Returns true if `value` is a memref in DTCM (#iree_codegen.workgroup_local).
bool isInDTCM(Value value) {
  auto type = dyn_cast<MemRefType>(value.getType());
  return type && isa_and_nonnull<IREE::Codegen::WorkgroupLocalMemoryAttr>(
                     type.getMemorySpace());
}

// Returns true if `value` is a memref whose innermost dimension (a row) is
// contiguous, so that vector loads and stores can access the rows.
bool hasContiguousRows(Value value) {
  auto type = cast<MemRefType>(value.getType());
  return type.getRank() > 0 && type.isLastDimUnitStride();
}

// Returns true if this pass lowers `copy`: a copy to or from DTCM, with
// contiguous rows and elements that fit in vectors.
bool isLowerableDTCMCopy(const MemRefCopy &copy) {
  // The other copies keep the IREE lowering.
  if (!isInDTCM(copy.source) && !isInDTCM(copy.target)) return false;
  // TODO: copies with strided rows (e.g. transposed tiles) stay scalar loops.
  if (!hasContiguousRows(copy.source) || !hasContiguousRows(copy.target)) {
    return false;
  }
  auto type = cast<MemRefType>(copy.source.getType());
  // A copy without elements needs no vector loop (and a vector cannot have 0
  // elements).
  if (llvm::is_contained(type.getShape(), 0)) return false;
  Type elementType = type.getElementType();
  return elementType.isIntOrFloat() &&
         elementType.getIntOrFloatBitWidth() % 8 == 0;
}

// Replaces `copy` with loops of vector loads and stores of `vectorSize`
// elements along the rows. The last vector of a row has a mask if the length
// of the rows is not a multiple of the vector size.
void lowerToVectorLoops(RewriterBase &rewriter, const MemRefCopy &copy,
                        int64_t vectorSize) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(copy.op);
  Location loc = copy.op->getLoc();
  auto type = cast<MemRefType>(copy.source.getType());
  int64_t rank = type.getRank();
  int64_t rowLength = type.getShape().back();
  // A short static row needs only one short power-of-2 vector.
  if (!ShapedType::isDynamic(rowLength)) {
    vectorSize = std::min<int64_t>(vectorSize, llvm::PowerOf2Ceil(rowLength));
  }
  bool needsMask =
      ShapedType::isDynamic(rowLength) || rowLength % vectorSize != 0;
  // The sizes of the dimensions, at the place of the copy.
  SmallVector<Value> sizes = getValueOrCreateConstantIndexOp(
      rewriter, loc, memref::getMixedSizes(rewriter, loc, copy.source));
  Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
  Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
  Value step = arith::ConstantIndexOp::create(rewriter, loc, vectorSize);
  // One loop for each dimension: the loop of the rows steps by vectors. The
  // canonicalizer removes the loops with one iteration.
  SmallVector<Value> indices;
  for (int64_t dim = 0; dim < rank; ++dim) {
    auto loop = scf::ForOp::create(rewriter, loc, zero, sizes[dim],
                                   dim == rank - 1 ? step : one);
    indices.push_back(loop.getInductionVar());
    rewriter.setInsertionPointToStart(loop.getBody());
  }
  auto vectorType = VectorType::get({vectorSize}, type.getElementType());
  // Full vectors need no mask.
  if (!needsMask) {
    Value data =
        vector::LoadOp::create(rewriter, loc, vectorType, copy.source, indices);
    vector::StoreOp::create(rewriter, loc, data, copy.target, indices);
    rewriter.eraseOp(copy.op);
    return;
  }
  // The last vector of a row covers the rest of the row. A single masked loop
  // handles both the full vectors and the tail to keep the ITCM code size
  // small.
  Value rest =
      arith::SubIOp::create(rewriter, loc, sizes.back(), indices.back());
  Value length = arith::MinSIOp::create(rewriter, loc, rest, step);
  Value mask = vector::CreateMaskOp::create(
      rewriter, loc, VectorType::get({vectorSize}, rewriter.getI1Type()),
      ValueRange{length});
  Value padding = arith::ConstantOp::create(
      rewriter, loc, cast<TypedAttr>(rewriter.getZeroAttr(vectorType)));
  Value data = vector::MaskedLoadOp::create(
      rewriter, loc, vectorType, copy.source, indices, mask, padding);
  vector::MaskedStoreOp::create(rewriter, loc, copy.target, indices, mask,
                                data);
  rewriter.eraseOp(copy.op);
}

struct CoralNPULowerDTCMCopiesPass
    : public impl::CoralNPULowerDTCMCopiesBase<CoralNPULowerDTCMCopiesPass> {
  using CoralNPULowerDTCMCopiesBase::CoralNPULowerDTCMCopiesBase;

  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();
    IRRewriter rewriter(&getContext());
    int64_t nativeVectorBytes = getNativeVectorBytes(funcOp);
    // The post-order walk allows the lowering to erase the copy and to insert
    // the loops before it. TODO(DMA): lower each copy to DMA transfers here
    // (see the file comment), and use the vector loops only as the fallback.
    funcOp.walk([&](Operation *op) {
      std::optional<MemRefCopy> copy = matchCopy(op);
      // Lower only the copies to and from DTCM that vectors can do.
      if (!copy || !isLowerableDTCMCopy(*copy)) return;
      Type elementType =
          cast<MemRefType>(copy->source.getType()).getElementType();
      // A vector fills the native vector size.
      int64_t vectorSize = std::max<int64_t>(
          1, nativeVectorBytes * 8 / elementType.getIntOrFloatBitWidth());
      lowerToVectorLoops(rewriter, *copy, vectorSize);
    });
  }
};

}  // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createCoralNPULowerDTCMCopiesPass() {
  return std::make_unique<CoralNPULowerDTCMCopiesPass>();
}

}  // namespace mlir::coralnpu_compiler
