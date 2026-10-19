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
// CoralNPU operand promotion: copies the tiles of a root op and of the ops
// fused with it to DTCM (#iree_codegen.workgroup_local memory). The buffers
// are the buffers of DTCMFootprintModel (CoralNPUTileSizeSelectionFootprint.h),
// which the DTCM tile-size selection counts: keep the two in sync.
//
// The copies:
// - An input tile goes to DTCM with a bufferization.materialize_in_destination
//   op. It is not a linalg op, so the vector-level tiling does not tile it and
//   does not fuse it into the loops of its users, and the vectorization does
//   not fold it away. It is a destination-style op, so the sizes of its result
//   are the sizes of the DTCM tensor.
// - A stored result is computed into a DTCM tensor (the output staging
//   buffer). The store op (parallel_insert_slice or store_to_buffer) then
//   copies it from DTCM to memory when it bufferizes.
// After bufferization, both kinds of copies are linalg.generic copies of whole
// tiles between memory and DTCM. CoralNPULowerDTCMCopies lowers them to vector
// loads and stores. TODO(DMA): lower them to DMA transfers there.
//
// Producers out of the tile loop: IREE fuses a producer into the tile loop
// only through a slice of its result, and the tiling makes no slice of an
// input that no tiled loop indexes (e.g. the RHS of a matmul when only M is
// tiled). Such a producer (e.g. a dequantization) stays before the loop, and
// the vector-level fusion later makes the inner loops read its input from
// memory, once for each vector tile that uses it. To copy such an input to
// DTCM like the inputs of the producers that IREE fuses, the pass first clones
// the elementwise producers out of the tile loop into it. (The footprint model
// counts these inputs too: keep the two in sync.) The clones compute the same
// values the same number of times: the tile loop has one tile per workgroup
// (`cache_parallel == distribution`), each workgroup already runs the code
// before the workgroup loop, and the vector-level fusion still fuses the
// clones into the inner loops.
//
// The DTCM buffers are static (the upper bounds of the tile sizes). They are
// at the start of the workgroup loop body, or of the function if there is no
// workgroup loop, and each tile uses a slice of its buffer. Thus:
// - Loop peeling clones the bodies of the inner loops, but not the buffers.
// - The canonicalizer does not make a dynamic bufferization.alloc_tensor
//   static (MLIR ReplaceStaticShapeDims drops the memory space when it does).
// - Bufferization does not copy the buffers: in each loop iteration, the tile
//   writes its buffer before it reads it.
// - The buffers become memref.alloc ops in the entry block of the function,
//   which iree-llvmcpu-assign-workgroup-local-memory requires.
//===----------------------------------------------------------------------===//

#include "compiler/Transforms/CoralNPUTileSizeSelectionUtils.h"
#include "compiler/Transforms/Passes.h"

// IREE:
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"

// MLIR:
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Remarks.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"

// LLVM:
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/DebugLog.h"

// Standard library:
#include <memory>
#include <optional>

#define DEBUG_TYPE "coralnpu-promote-operands"

using namespace mlir;
using namespace mlir::iree_compiler;

namespace mlir::coralnpu_compiler {

#define GEN_PASS_DEF_CORALNPUPROMOTEOPERANDS
#include "compiler/Transforms/Passes.h.inc"

namespace {

// Reports a tensor that the promotion keeps in memory, which can degrade the
// performance: a missed remark named `name` with the reason `reason`, and
// debug output. A normal compilation does not print them.
void reportNotPromoted(Location loc, StringRef name, StringRef reason) {
  LDBG() << name << ": " << reason;
  remark::missed(loc, remark::RemarkOpts::name(name).category(DEBUG_TYPE))
      << remark::reason("{0}", reason);
}

// Returns true if `value` is a tensor that the dispatch loads from a binding (a
// load_from_buffer, or a dispatch.tensor.load of a region of a binding), or a
// tile of such a tensor (an extract_slice, maybe of other extract_slices).
bool isBindingTile(Value value) {
  // Look through the slices.
  while (auto slice = value.getDefiningOp<tensor::ExtractSliceOp>()) {
    value = slice.getSource();
  }
  return isa_and_nonnull<IREE::Codegen::LoadFromBufferOp,
                         IREE::TensorExt::DispatchTensorLoadOp>(
      value.getDefiningOp());
}

// Returns the operand of `op` that holds the tensor that `op` stores: the
// source of a parallel_insert_slice, store_to_buffer or dispatch.tensor.store.
// Returns null for other ops.
OpOperand *getStoredOperand(Operation *op) {
  return TypeSwitch<Operation *, OpOperand *>(op)
      .Case([](tensor::ParallelInsertSliceOp insert) {
        return &insert.getSourceMutable();
      })
      .Case([](IREE::Codegen::StoreToBufferOp store) {
        return &store.getTensorMutable();
      })
      .Case([](IREE::TensorExt::DispatchTensorStoreOp store) {
        return &store.getValueMutable();
      })
      .Default([](Operation *) -> OpOperand * { return nullptr; });
}

// Returns true if `forallOp` distributes its iterations to workgroups.
bool isWorkgroupForall(scf::ForallOp forallOp) {
  std::optional<ArrayAttr> mapping = forallOp.getMapping();
  return mapping &&
         llvm::any_of(*mapping,
                      llvm::IsaPred<IREE::Codegen::WorkgroupMappingAttr>);
}

// Returns the block that gets the DTCM buffers of the tiles of `rootOp`: the
// body of the workgroup loop around `rootOp`, or else the entry block of
// `funcOp`.
Block *getAllocBlock(linalg::LinalgOp rootOp, FunctionOpInterface funcOp) {
  // Look for the workgroup loop in the loops around the root op.
  for (auto forallOp = rootOp->getParentOfType<scf::ForallOp>(); forallOp;
       forallOp = forallOp->getParentOfType<scf::ForallOp>()) {
    if (isWorkgroupForall(forallOp)) return forallOp.getBody();
  }
  return &funcOp.getFunctionBody().front();
}

// Clones into a tile loop the elementwise producers out of it whose inputs are
// binding tiles (see the file comment). The clones go at the start of the loop
// body, in the order of creation.
class ProducerCloner {
 public:
  ProducerCloner(RewriterBase &rewriter, scf::ForallOp tileLoop)
      : rewriter(rewriter),
        tileRegion(tileLoop.getRegion()),
        insertionAnchor(&tileLoop.getBody()->front()) {}

  // Returns the value for the ops in the tile loop to use instead of `value`:
  // the result of a clone of the producer of `value` if the producer is an
  // elementwise linalg op out of the loop that reads a binding tile (directly,
  // or through other such producers). Else returns `value`.
  Value getValueInLoop(Value value) {
    // Each producer gets one clone.
    if (Value clone = clones.lookup(value)) return clone;
    auto producer = value.getDefiningOp<linalg::LinalgOp>();
    // Only an elementwise linalg op with one result, out of the loop, gets a
    // clone. The clone keeps the init of the producer, so a producer that
    // reads its init (which the promotion does not copy) gets no clone.
    if (!producer || producer->getNumResults() != 1 ||
        tileRegion.isAncestor(producer->getParentRegion()) ||
        !linalg::isElementwise(producer) ||
        producer.payloadUsesValueFromOperand(producer.getDpsInitOperand(0))) {
      return value;
    }
    // The inputs of the clone: the values in the loop for the inputs of the
    // producer.
    SmallVector<Value> inputs;
    bool readsBindingTile = false;
    for (Value input : producer.getDpsInputs()) {
      Value inputInLoop = getValueInLoop(input);
      readsBindingTile |= inputInLoop != input || isBindingTile(input);
      inputs.push_back(inputInLoop);
    }
    // A clone helps only if the promotion can copy an input of it to DTCM.
    if (!readsBindingTile) {
      clones[value] = value;
      return value;
    }
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(insertionAnchor);
    auto clone = cast<linalg::LinalgOp>(rewriter.clone(*producer));
    // The clone reads the values in the loop.
    for (auto [operand, input] :
         llvm::zip_equal(clone.getDpsInputOperands(), inputs)) {
      operand->set(input);
    }
    clonedProducers.push_back(producer);
    Value result = clone->getResult(0);
    clones[value] = result;
    return result;
  }

  // Erases the cloned producers that have no use left.
  void eraseDeadProducers() {
    // Erase the users of a producer before the producer.
    for (Operation *producer : llvm::reverse(clonedProducers)) {
      if (producer->use_empty()) rewriter.eraseOp(producer);
    }
  }

 private:
  RewriterBase &rewriter;
  Region &tileRegion;
  // The clones go before this op (the first op of the loop body), so that a
  // clone comes after the clones of its producers.
  Operation *insertionAnchor;
  // The value in the loop for each visited producer result (`value` itself if
  // the producer is not cloned).
  llvm::DenseMap<Value, Value> clones;
  // The cloned producers, each after its own cloned producers.
  SmallVector<Operation *> clonedProducers;
};

// Makes the linalg ops in `tileLoop` read clones of the elementwise producers
// out of the loop whose inputs are binding tiles (see ProducerCloner), so that
// the promotion copies these inputs to DTCM.
void cloneProducersIntoTileLoop(RewriterBase &rewriter,
                                scf::ForallOp tileLoop) {
  SmallVector<linalg::LinalgOp> ops;
  tileLoop.getRegion().walk([&](linalg::LinalgOp op) { ops.push_back(op); });
  ProducerCloner cloner(rewriter, tileLoop);
  // Replace the inputs of the ops of the loop that come from such producers.
  for (linalg::LinalgOp op : ops) {
    for (OpOperand *input : op.getDpsInputOperands()) {
      Value inputInLoop = cloner.getValueInLoop(input->get());
      // The other inputs do not change.
      if (inputInLoop == input->get()) continue;
      rewriter.modifyOpInPlace(op, [&]() { input->set(inputInLoop); });
    }
  }
  cloner.eraseDeadProducers();
}

// Returns the sizes of `value` (a ranked tensor): the size operands of a slice,
// or else tensor.dim ops for the dynamic sizes.
SmallVector<OpFoldResult> getSizes(OpBuilder &builder, Location loc,
                                   Value value) {
  auto slice = value.getDefiningOp<tensor::ExtractSliceOp>();
  // The sizes of a slice that keeps all dimensions are its result sizes.
  if (slice && slice.getSourceType().getRank() == slice.getType().getRank()) {
    return slice.getMixedSizes();
  }
  return tensor::getMixedSizes(builder, loc, value);
}

// Returns the shape of `value` (a ranked tensor) with the constant upper bounds
// of its dynamic sizes. Returns failure if a dynamic size has no constant
// upper bound.
FailureOr<SmallVector<int64_t>> getUpperBoundShape(Value value) {
  auto type = cast<RankedTensorType>(value.getType());
  SmallVector<int64_t> shape(type.getShape());
  // Replace each dynamic size with its constant upper bound.
  for (int64_t dim = 0, rank = type.getRank(); dim < rank; ++dim) {
    // A static size is its own bound.
    if (!ShapedType::isDynamic(shape[dim])) continue;
    FailureOr<int64_t> bound = ValueBoundsConstraintSet::computeConstantBound(
        presburger::BoundType::UB,
        ValueBoundsConstraintSet::Variable(value, dim),
        /*stopCondition=*/nullptr, ValueBoundsOptions{/*closedUB=*/true});
    // A size without a bound (e.g. from a push constant) has no buffer.
    if (failed(bound)) return failure();
    shape[dim] = *bound;
  }
  return shape;
}

// Creates the operations of the promotion of the tiles.
class Promoter {
 public:
  // The DTCM buffers go at the start of `allocBlock`.
  Promoter(RewriterBase &rewriter, Block *allocBlock)
      : rewriter(rewriter),
        allocBlock(allocBlock),
        allocAnchor(&allocBlock->front()),
        localMemory(IREE::Codegen::WorkgroupLocalMemoryAttr::get(
            rewriter.getContext())) {}

  // Returns a DTCM tensor with the type and the sizes of `like`, at the
  // insertion point. The DTCM buffer is static, at the start of the alloc
  // block, with the upper bounds of the sizes of `like`; if `like` has dynamic
  // sizes, the result is a slice of the buffer. Returns failure if a dynamic
  // size has no constant upper bound.
  FailureOr<Value> createLocalTensor(Location loc, Value like) {
    auto type = cast<RankedTensorType>(like.getType());
    FailureOr<SmallVector<int64_t>> shape = getUpperBoundShape(like);
    if (failed(shape)) return failure();
    Value buffer =
        createBuffer(loc, cast<RankedTensorType>(type.clone(*shape)));
    // A static tensor is the whole buffer.
    if (type.hasStaticShape()) return buffer;
    int64_t rank = type.getRank();
    SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));
    return tensor::ExtractSliceOp::create(rewriter, loc, type, buffer, offsets,
                                          getSizes(rewriter, loc, like),
                                          strides)
        .getResult();
  }

  // Copies `value` to a new DTCM tensor with a materialize_in_destination op.
  // Returns the copy, or failure if the DTCM tensor has no static bounds.
  // TODO(DMA): the copy becomes a linalg.generic copy when it bufferizes, and
  // CoralNPULowerDTCMCopies lowers it: lower it to a DMA transfer there.
  FailureOr<Value> copyToLocalMemory(Value value) {
    OpBuilder::InsertionGuard guard(rewriter);
    setInsertionPointAfterDefinition(value);
    Location loc = value.getLoc();
    FailureOr<Value> local = createLocalTensor(loc, value);
    if (failed(local)) return failure();
    return bufferization::MaterializeInDestinationOp::create(rewriter, loc,
                                                             value, *local)
        .getResult();
  }

  // Makes `init` (a DPS init of a linalg op) a DTCM tensor. A fill writes into
  // a new DTCM tensor; another init that the op reads is copied to DTCM.
  // Returns failure if `init` is not a tensor with static bounds in the alloc
  // block.
  LogicalResult makeLocalInit(OpOperand &init, bool readsInit) {
    Operation *owner = init.getOwner();
    // Only a tensor init of an op in the alloc block can use a DTCM buffer.
    if (!isa<RankedTensorType>(init.get().getType()) ||
        !allocBlock->getParent()->isAncestor(owner->getParentRegion())) {
      return failure();
    }

    // A fill only needs a destination: fill a DTCM tensor instead.
    if (auto fill = init.get().getDefiningOp<linalg::FillOp>()) {
      OpBuilder::InsertionGuard guard(rewriter);
      // IREE's tile-and-fuse passes normally fuse the fill into the parallel
      // tile loop, which is inside the alloc block. If a fill was left outside
      // the alloc block or has other users, clone it at `owner` so it can use
      // the DTCM buffer in the alloc block without breaking SSA dominance or
      // affecting other users.
      bool inAllocRegion =
          allocBlock->getParent()->isAncestor(fill->getParentRegion());
      rewriter.setInsertionPoint(inAllocRegion ? fill.getOperation() : owner);
      OpOperand &fillInit = fill.getDpsInitsMutable()[0];
      FailureOr<Value> local = createLocalTensor(fill.getLoc(), fillInit.get());
      if (failed(local)) return failure();
      if (!fill->hasOneUse() || !inAllocRegion) {
        auto newFill = cast<linalg::FillOp>(rewriter.clone(*fill));
        newFill.getDpsInitsMutable()[0].set(*local);
        rewriter.modifyOpInPlace(owner,
                                 [&]() { init.set(newFill.getResult(0)); });
        if (fill->use_empty()) rewriter.eraseOp(fill);
        return success();
      }
      rewriter.modifyOpInPlace(fill, [&]() { fillInit.set(*local); });
      return success();
    }

    // An init that the op only writes needs no copy.
    if (!readsInit) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(owner);
      FailureOr<Value> local = createLocalTensor(owner->getLoc(), init.get());
      if (failed(local)) return failure();
      rewriter.modifyOpInPlace(owner, [&]() { init.set(*local); });
      return success();
    }

    FailureOr<Value> local = copyToLocalMemory(init.get());
    if (failed(local)) return failure();
    rewriter.modifyOpInPlace(owner, [&]() { init.set(*local); });
    return success();
  }

 private:
  // Creates a DTCM buffer of the static `type` at the start of the alloc block
  // (before allocAnchor, after the ops that the promotion put there before).
  Value createBuffer(Location loc, RankedTensorType type) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(allocAnchor);
    auto alloc = bufferization::AllocTensorOp::create(
        rewriter, loc, type, /*dynamicSizes=*/ValueRange());
    alloc.setMemorySpaceAttr(localMemory);
    return alloc.getResult();
  }

  // Sets the insertion point right after the definition of `value`. If `value`
  // is an argument of the alloc block or comes from outside of the block, sets
  // it at the start of the block (before allocAnchor; then a copy runs once in
  // each workgroup). The buffers of later calls then come after such a copy.
  // This order has no effect: bufferization moves the buffers to the entry
  // block of the function.
  void setInsertionPointAfterDefinition(Value value) {
    Block *block = value.getParentBlock();
    bool isAllocBlockArgument =
        isa<BlockArgument>(value) && block == allocBlock;
    // The values from outside are available at the start of the block.
    if (isAllocBlockArgument ||
        !allocBlock->getParent()->isAncestor(block->getParent())) {
      rewriter.setInsertionPoint(allocAnchor);
      return;
    }
    rewriter.setInsertionPointAfterValue(value);
  }

  RewriterBase &rewriter;
  Block *allocBlock;
  // The first op of the alloc block before the promotion. The buffers, and the
  // copies of the values from outside of the block, go before it, in the order
  // of creation.
  Operation *allocAnchor;
  Attribute localMemory;
};

// Returns the value that a chain of loops in `allocBlock` gives to the root op
// as `init`: the init of the outermost loop if `init` is a loop-carried value
// (split K), else `init` itself. Sets `outerLoop` to the outermost such loop,
// or null.
OpOperand *getLoopInit(OpOperand *init, Block *allocBlock,
                       scf::ForOp &outerLoop) {
  outerLoop = nullptr;
  Region *allocRegion = allocBlock->getParent();
  // Go out through the loops in the alloc block that carry the value.
  while (auto bbArg = dyn_cast<BlockArgument>(init->get())) {
    auto loop = dyn_cast<scf::ForOp>(bbArg.getOwner()->getParentOp());
    // Stop at a block argument that is not an iter_arg of an scf.for in the
    // alloc block.
    if (!loop || bbArg.getArgNumber() < loop.getNumInductionVars() ||
        !allocRegion->isAncestor(loop->getParentRegion())) {
      break;
    }
    outerLoop = loop;
    init = loop.getTiedLoopInit(bbArg);
  }
  return init;
}

// Returns the ops that store the tiles of the root op and of the ops fused
// with it: the parallel_insert_slice ops of `tileLoop`, or the store ops of
// `funcOp` if there is no tile loop.
SmallVector<Operation *> collectStores(scf::ForallOp tileLoop,
                                       FunctionOpInterface funcOp) {
  SmallVector<Operation *> stores;
  // A tile loop stores the tiles in its terminator.
  if (tileLoop) {
    for (Operation &op : tileLoop.getTerminator().getYieldingOps()) {
      if (isa<tensor::ParallelInsertSliceOp>(op)) stores.push_back(&op);
    }
    return stores;
  }
  // Without a tile loop, the store ops write the whole results to the
  // bindings.
  funcOp.walk([&](Operation *op) {
    if (isa<IREE::Codegen::StoreToBufferOp,
            IREE::TensorExt::DispatchTensorStoreOp>(op)) {
      stores.push_back(op);
    }
  });
  return stores;
}

struct CoralNPUPromoteOperandsPass
    : public impl::CoralNPUPromoteOperandsBase<CoralNPUPromoteOperandsPass> {
  using CoralNPUPromoteOperandsBase::CoralNPUPromoteOperandsBase;

  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();

    // The root op is the linalg op whose lowering config asks for promotion.
    // CoralNPUTileSizeSelectionDTCM sets promote_operands only on the root op,
    // so the walk stops at the first such op. The operand indices in
    // promote_operands are for compatibility with IREE's attribute format (see
    // kPromoteOperandsName): this pass promotes all external inputs and stored
    // outputs of the fused tile, which may belong to fused producers or
    // consumers rather than the root op itself.
    linalg::LinalgOp rootOp;
    funcOp.walk([&](linalg::LinalgOp op) {
      if (hasPromoteOperands(getLoweringConfig(op))) {
        rootOp = op;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    // Nothing to promote.
    if (!rootOp || rootOp->getNumResults() == 0) return;
    // The promotion supports one accumulator.
    if (rootOp->getNumResults() > 1) {
      reportNotPromoted(
          rootOp->getLoc(), "MultiResultRoot",
          "the root op has more than one result; the operands stay in memory");
      return;
    }

    // The innermost parallel loop around the root op writes the tiles. Without
    // such a loop, the root op computes the whole tensor.
    auto tileLoop = rootOp->getParentOfType<scf::ForallOp>();
    Region &tileRegion =
        tileLoop ? tileLoop.getRegion() : funcOp.getFunctionBody();

    IRRewriter rewriter(&getContext());
    // Bring the producers that IREE left out of the tile loop into it, so that
    // their inputs get DTCM copies too.
    if (tileLoop) cloneProducersIntoTileLoop(rewriter, tileLoop);

    // The ops that the promotion changes, collected before the changes.
    SmallVector<linalg::LinalgOp> fusedOps;
    tileRegion.walk([&](linalg::LinalgOp op) { fusedOps.push_back(op); });
    SmallVector<Operation *> stores = collectStores(tileLoop, funcOp);

    Block *allocBlock = getAllocBlock(rootOp, funcOp);
    Promoter promoter(rewriter, allocBlock);
    Value rootResult = promoteRootInit(promoter, rootOp, allocBlock);
    promoteInputs(rewriter, promoter, fusedOps);
    promoteOutputs(promoter, rootResult, stores);
  }

 private:
  // Gives the root op a DTCM accumulator if the tile splits K (the accumulator
  // stays in DTCM across the K chunks), or if the root result is stored (then
  // the accumulator is also the output staging buffer). Returns the root result
  // (after any K loops) when handled here, or null otherwise.
  Value promoteRootInit(Promoter &promoter, linalg::LinalgOp rootOp,
                        Block *allocBlock) {
    scf::ForOp outerLoop;
    OpOperand *init =
        getLoopInit(rootOp.getDpsInitOperand(0), allocBlock, outerLoop);
    Value rootResult =
        outerLoop ? outerLoop.getTiedLoopResult(init) : rootOp->getResult(0);
    bool isStored = llvm::any_of(rootResult.getUses(), [](OpOperand &use) {
      return getStoredOperand(use.getOwner()) == &use;
    });
    // In-register accumulators need no DTCM buffer.
    if (!outerLoop && !isStored) return Value();
    // The root op reads its init (it accumulates).
    if (failed(promoter.makeLocalInit(*init, /*readsInit=*/true))) {
      reportNotPromoted(
          rootOp->getLoc(), "UnboundedAccumulator",
          "the accumulator has no static size bound; it stays in memory");
    }
    return rootResult;
  }

  // Copies the binding tiles that the fused ops read to DTCM.
  void promoteInputs(RewriterBase &rewriter, Promoter &promoter,
                     ArrayRef<linalg::LinalgOp> fusedOps) {
    // The uses of each binding tile, in the order of the fused ops.
    llvm::MapVector<Value, SmallVector<OpOperand *>> tiles;
    for (linalg::LinalgOp op : fusedOps) {
      for (OpOperand *input : op.getDpsInputOperands()) {
        // Scalars live in registers; other tensors are computed by fused ops.
        if (isBindingTile(input->get())) tiles[input->get()].push_back(input);
      }
    }
    // One copy for each tile, which all fused ops read.
    for (auto &[tile, uses] : tiles) {
      FailureOr<Value> local = promoter.copyToLocalMemory(tile);
      // Without static bounds, the fused ops read the tile from memory.
      if (failed(local)) {
        reportNotPromoted(
            tile.getLoc(), "UnboundedTile",
            "the tile has no static size bound; it stays in memory");
        continue;
      }
      // The fused ops read the DTCM copy.
      for (OpOperand *use : uses) {
        rewriter.modifyOpInPlace(use->getOwner(), [&]() { use->set(*local); });
      }
    }
  }

  // Makes each stored result of the root op or of a fused consumer go through
  // an output staging buffer in DTCM: the op that computes the result writes it
  // into DTCM, and the store op copies it to memory when it bufferizes.
  // TODO(DMA): the copies of the stores become linalg.generic copies when they
  // bufferize, and CoralNPULowerDTCMCopies lowers them: lower them to DMA
  // transfers there.
  void promoteOutputs(Promoter &promoter, Value rootResult,
                      ArrayRef<Operation *> stores) {
    // The consumer results that write into DTCM already.
    llvm::DenseSet<Value> stagedResults;
    // Stage each stored result in DTCM.
    for (Operation *store : stores) {
      Value source = getStoredOperand(store)->get();
      // promoteRootInit gave the root op (or its K loop) a DTCM accumulator,
      // or reported that it could not.
      if (source == rootResult) continue;
      // A result that more than one op stores needs one staging buffer.
      if (stagedResults.contains(source)) continue;
      // Other results come from a consumer, which must write into DTCM.
      if (failed(makeLocalResult(promoter, source))) {
        reportNotPromoted(store->getLoc(), "UnstagedResult",
                          "the stored result does not come from a fused linalg "
                          "op with static bounds; it is not staged in DTCM");
        continue;
      }
      stagedResults.insert(source);
    }
  }

  // Makes the fused linalg op that computes `result` write `result` into DTCM.
  // Returns failure if `result` does not come from a linalg op, or has no
  // static bounds.
  LogicalResult makeLocalResult(Promoter &promoter, Value result) {
    auto opResult = dyn_cast<OpResult>(result);
    auto consumer =
        opResult ? dyn_cast<linalg::LinalgOp>(opResult.getOwner()) : nullptr;
    // Only a linalg op can write into DTCM.
    if (!consumer) return failure();
    OpOperand *init = consumer.getDpsInitOperand(opResult.getResultNumber());
    return promoter.makeLocalInit(*init,
                                  consumer.payloadUsesValueFromOperand(init));
  }
};

}  // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createCoralNPUPromoteOperandsPass() {
  return std::make_unique<CoralNPUPromoteOperandsPass>();
}

}  // namespace mlir::coralnpu_compiler
