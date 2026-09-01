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
#include "compiler/Transforms/Passes.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

//===----------------------------------------------------------------------===//
// CoralNPU Matrix Codegen Pass
//
// Lowers 2D vector.contract to Zvt matrix instructions (msetmtype, vtzero,
// vtmms.tvv / vtfmm.tvv, vtmv.v.t).
//
// Supports FP32->FP32 and INT8->INT32 in 16x16, 16x32, 32x16 and 32x32 tile
// shapes. A contraction yielded by an scf.for reduction replaces the whole
// loop with one inline-asm K loop holding the accumulators in mt0..mt12.
//
// The tiles are always vtzero'd first, so vector.contract's `acc` operand is
// discarded. That matches the IREE-generated form (pre-zeroed destination);
// a non-zero `acc` cannot be honoured since no tile load is wired up.
//===----------------------------------------------------------------------===//

namespace mlir::coralnpu_compiler {

#define GEN_PASS_DEF_CORALNPUMATRIXCODEGEN
#include "compiler/Transforms/Passes.h.inc"

namespace {

// A contraction plus its destination (base memref + row/col indices) and the
// enclosing K-reduction scf.for loop, if any.
struct AccumulatorChain {
  vector::ContractionOp contractOp;
  Value baseC, rowC, colC;
  scf::ForOp loop = nullptr;
  SmallVector<Operation*> opsToErase;
};

Value traceToSourceValue(Value val) {
  while (Operation* def = val.getDefiningOp()) {
    if (isa<arith::ExtFOp, arith::ExtSIOp, arith::ExtUIOp, vector::BroadcastOp,
            vector::ShapeCastOp, vector::ExtractStridedSliceOp>(def)) {
      val = def->getOperand(0);
    } else if (auto fe = dyn_cast<vector::FromElementsOp>(def)) {
      if (fe.getElements().empty()) break;
      auto ex = fe.getElements().front().getDefiningOp<vector::ExtractOp>();
      if (!ex) break;
      val = ex.getSource();
    } else {
      break;
    }
  }
  return val;
}

Type traceToSourceElementType(Value val) {
  Type t = traceToSourceValue(val).getType();
  return isa<VectorType>(t) ? cast<VectorType>(t).getElementType() : t;
}

// Peels memref.subview chains so two views of one buffer compare equal.
Value getRootBuffer(Value v) {
  while (auto sv = v ? v.getDefiningOp<memref::SubViewOp>() : nullptr)
    v = sv.getSource();
  return v;
}

bool isSupportedMatrixContraction(vector::ContractionOp op) {
  auto lhsTy = dyn_cast<VectorType>(op.getLhs().getType());
  auto rhsTy = dyn_cast<VectorType>(op.getRhs().getType());
  auto accTy = dyn_cast<VectorType>(op.getAcc().getType());
  if (!lhsTy || !rhsTy || !accTy || accTy.getRank() != 2) return false;
  int64_t m = accTy.getDimSize(0), n = accTy.getDimSize(1);
  if ((m != 16 && m != 32) || (n != 16 && n != 32)) return false;
  if (lhsTy.getNumElements() != m || rhsTy.getNumElements() != n) return false;
  Type lhsElTy = traceToSourceElementType(op.getLhs());
  Type rhsElTy = traceToSourceElementType(op.getRhs());
  Type accElTy = accTy.getElementType();
  if (lhsElTy != rhsElTy) return false;
  if (!(lhsElTy.isF32() && accElTy.isF32()) &&
      !(lhsElTy.isInteger(8) && accElTy.isInteger(32))) {
    return false;
  }
  auto maps = op.getIndexingMapsArray();
  auto iters = op.getIteratorTypes().getValue();
  return maps.size() == 3 && iters.size() >= 3 &&
         maps[2].getNumResults() == 2 &&
         vector::isParallelIterator(iters[maps[2].getDimPosition(0)]) &&
         vector::isParallelIterator(iters[maps[2].getDimPosition(1)]);
}

void createInlineAsm(OpBuilder& b, Location loc, StringRef asmStr,
                     StringRef constraints, ValueRange operands) {
  LLVM::InlineAsmOp::create(
      b, loc, TypeRange{}, operands, b.getStringAttr(asmStr),
      b.getStringAttr(constraints), /*has_side_effects=*/b.getUnitAttr(),
      /*is_align_stack=*/nullptr, /*tail_call_kind=*/nullptr,
      LLVM::AsmDialectAttr::get(b.getContext(), LLVM::AsmDialect::AD_ATT),
      /*operand_attrs=*/b.getArrayAttr({}));
}

// The Zvt matrix unit operates on fixed 16x16 tiles.
constexpr int64_t kTileDim = 16;

// Minimum element count of the scalable type used for inline-asm "^vr"
// (vector register) operands.
constexpr int64_t kScalableMinElems = 8;

Value castToScalable(OpBuilder& b, Location loc, Value val) {
  VectorType ty = cast<VectorType>(val.getType());
  Value flat = vector::ShapeCastOp::create(
      b, loc, VectorType::get({ty.getNumElements()}, ty.getElementType()), val);
  VectorType scalableTy =
      VectorType::get({kScalableMinElems}, ty.getElementType(), {true});
  Value zero =
      LLVM::ConstantOp::create(b, loc, b.getI64Type(), b.getI64IntegerAttr(0));
  return LLVM::CallIntrinsicOp::create(
             b, loc, static_cast<Type>(scalableTy),
             b.getStringAttr("llvm.vector.insert"),
             ValueRange{LLVM::UndefOp::create(b, loc, scalableTy), flat, zero})
      .getResult(0);
}

// Returns the `tileIdx`-th contiguous kTileDim-element slice of `val`.
Value extractSubVector(OpBuilder& b, Location loc, Value val, int64_t tileIdx) {
  auto vecTy = cast<VectorType>(val.getType());
  if (vecTy.getNumElements() == kTileDim) return val;
  Value mat = vector::ShapeCastOp::create(
      b, loc,
      VectorType::get({vecTy.getNumElements() / kTileDim, kTileDim},
                      vecTy.getElementType()),
      val);
  return vector::ExtractOp::create(b, loc, mat, ArrayRef<int64_t>{tileIdx});
}

std::pair<Value, Value> extractPtrAndStride(OpBuilder& b, Location loc,
                                            Value memref, Value row, Value col,
                                            int64_t elemBytes) {
  auto s = memref::ExtractStridedMetadataOp::create(b, loc, memref);
  auto strides = s.getStrides();
  Value stride0 = strides.size() >= 2
                      ? Value(strides[strides.size() - 2])
                      : Value(arith::ConstantIndexOp::create(b, loc, 1));
  Value base =
      memref::ExtractAlignedPointerAsIndexOp::create(b, loc, s.getBaseBuffer());
  Value bytes = arith::ConstantIndexOp::create(b, loc, elemBytes);
  auto add = [&](Value x, Value y) {
    return arith::AddIOp::create(b, loc, x, y);
  };
  auto mul = [&](Value x, Value y) {
    return arith::MulIOp::create(b, loc, x, y);
  };
  Value off = add(s.getOffset(), add(mul(row, stride0), col));
  Type i32 = b.getI32Type();
  return {arith::IndexCastOp::create(b, loc, i32, add(base, mul(off, bytes))),
          arith::IndexCastOp::create(b, loc, i32, mul(stride0, bytes))};
}

// Zvt accumulator tiles are always 32-bit (f32 or i32).
constexpr int64_t kAccElemBytes = 4;

// msetmtype operands (decode: third_party/coralnpu .../design/RvvFrontEnd.sv).
// mtype (rs1): tm = bits[23:10], tk = bits[7:5], mtwiden = bits[1:0]. Always
// TM=16, TK=1; TWIDEN is 3 for i8 (8->32) and 1 for f32.
constexpr int64_t kMtypeI8 = (16 << 10) | (1 << 5) | 3;
constexpr int64_t kMtypeF32 = (16 << 10) | (1 << 5) | 1;
// vtype (rs2): SEW = bits[5:3], LMUL = bits[2:0], altfmt = bit 8. altfmt makes
// the *second* multiply operand signed (vtmms.tvv encodes only operand A's
// signedness), so signed INT8 GEMM needs it; FP32 leaves it clear.
constexpr int64_t kVtypeI8 = (0 << 3) | 0 | (1 << 8);  // SEW8, LMUL1, altfmt
constexpr int64_t kVtypeF32 = (2 << 3) | 2;            // SEW32, LMUL4

// Preamble for every Zvt asm block: program mtype/vtype, then TN=16. Per block,
// not once per function: any vset* clears mtype and vtype.altfmt
// (RvvFrontEnd.sv) and LLVM may insert one between blocks. With mtwiden != 0
// msetmtype derives LMUL/ta/ma from SEW and programs TM/TK (replacing vsetivli
// + msettm) and zeroes vl, hence the msettn. Clobbers t5 and t6.
std::string emitTileConfigAsm(bool isI8) {
  return llvm::formatv(
      "li t5, {0}\n\tli t6, {1}\n\tmsetmtype t5, t6\n\t"
      "li t6, 16\n\tmsettn zero, t6\n\t",
      isI8 ? kMtypeI8 : kMtypeF32, isI8 ? kVtypeI8 : kVtypeF32);
}

// Drains accumulator tiles mt0..mt12 to memory via vtmv.v.t + vse32.v.
// `ptrReg`/`strideReg` name the asm operands holding the pointer and stride.
std::string emitTileWritebackAsm(int64_t numTilesM, int64_t numTilesN,
                                 StringRef ptrReg, StringRef strideReg) {
  std::string asmStr = emitTileConfigAsm(/*isI8=*/false) +
                       llvm::formatv("mv t5, {0}\n\t{1}", strideReg,
                                     numTilesM > 1 ? "slli t1, t5, 4\n\t" : "")
                           .str();
  const char* lui[] = {"li t0, 0\n\t", "lui t0, 0x20000\n\t",
                       "lui t0, 0x40000\n\t", "lui t0, 0x60000\n\t"};
  int label = 2;
  for (int64_t m = 0; m < numTilesM; ++m) {
    for (int64_t n = 0; n < numTilesN; ++n) {
      asmStr += m ? llvm::formatv("add t4, {0}, t1\n\t", ptrReg)
                  : llvm::formatv("mv t4, {0}\n\t", ptrReg);
      if (n)
        asmStr += llvm::formatv("addi t4, t4, {0}\n\t", n * 16 * kAccElemBytes);
      asmStr += llvm::formatv(
          "{0}li t6, 16\n\t{1}:\n\tvtmv.v.t v0, t0\n\t"
          "vse32.v v0, (t4)\n\tadd t4, t4, t5\n\t"
          "addi t0, t0, 1\n\taddi t6, t6, -1\n\tbnez t6, {1}b\n\t",
          lui[m * 2 + n], label++);
    }
  }
  return asmStr;
}

// vtfmm/vtmms accumulate into mt in place, so tiles must be cleared before the
// first multiply. Accumulators are 32-bit in both modes, so this always runs
// under the f32 (SEW32/m4) config.
std::string emitTileZeroAsm(int64_t numTilesM, int64_t numTilesN) {
  std::string asmStr;
  for (int64_t m = 0; m < numTilesM; ++m)
    for (int64_t n = 0; n < numTilesN; ++n)
      asmStr += llvm::formatv("vtzero mt{0}\n\t", (m * 2 + n) * 4);
  return asmStr;
}

void emitInlineFusedGEMM(OpBuilder& b, Location loc, Value baseA, Value rowA,
                         Value colA, Value baseB, Value rowB, Value colB,
                         Value baseC, Value rowC, Value colC, int64_t numTilesM,
                         int64_t numTilesN, int64_t numKSteps, Type inputElTy,
                         bool isTransposedA) {
  bool isI8 = inputElTy.isInteger(8);
  int64_t inBytes = isI8 ? 1 : 4;
  auto [ptrA, strideA] =
      extractPtrAndStride(b, loc, baseA, rowA, colA, inBytes);
  auto [ptrB, strideB] =
      extractPtrAndStride(b, loc, baseB, rowB, colB, inBytes);
  auto [ptrC, strideC] =
      extractPtrAndStride(b, loc, baseC, rowC, colC, kAccElemBytes);

  // Both config blocks precede the t0-t3 setup: emitTileConfigAsm needs t5/t6
  // as scratch.
  std::string asmStr =
      emitTileConfigAsm(/*isI8=*/false) + emitTileZeroAsm(numTilesM, numTilesN);
  if (isI8) asmStr += emitTileConfigAsm(/*isI8=*/true);
  asmStr += "mv t0, $0\n\tmv t1, $1\n\tmv t2, $2\n\tmv t3, $3\n\t";

  if (numTilesM > 1) {
    if (isTransposedA)
      asmStr += llvm::formatv("addi t5, t0, {0}\n\t", 16 * inBytes);
    else
      asmStr += "slli t5, t1, 4\n\tadd t5, t0, t5\n\t";
  }
  if (numTilesN > 1)
    asmStr += llvm::formatv("addi t4, t2, {0}\n\t", 16 * inBytes);

  asmStr += llvm::formatv("li t6, {0}\n\t1:\n\t", numKSteps);

  StringRef loadA = isTransposedA ? (isI8 ? "vle8.v" : "vle32.v")
                                  : (isI8 ? "vlse8.v" : "vlse32.v");
  StringRef loadB = isI8 ? "vle8.v" : "vle32.v";
  StringRef opMul = isI8 ? "vtmms.tvv" : "vtfmm.tvv";

  auto appendLoadA = [&](StringRef reg, StringRef base) {
    asmStr += isTransposedA
                  ? llvm::formatv("{0} {1}, ({2})\n\t", loadA, reg, base)
                  : llvm::formatv("{0} {1}, ({2}), t1\n\t", loadA, reg, base);
  };
  appendLoadA("v4", "t0");
  if (numTilesM > 1) appendLoadA("v24", "t5");
  asmStr += llvm::formatv("{0} v8, (t2)\n\t", loadB);
  if (numTilesN > 1) asmStr += llvm::formatv("{0} v12, (t4)\n\t", loadB);

  for (int64_t m = 0; m < numTilesM; ++m)
    for (int64_t n = 0; n < numTilesN; ++n)
      asmStr += llvm::formatv("{0} mt{1}, {2}, {3}\n\t", opMul, (m * 2 + n) * 4,
                              m ? "v24" : "v4", n ? "v12" : "v8");

  if (isTransposedA) {
    asmStr += (numTilesM > 1) ? "add t0, t0, t1\n\tadd t5, t5, t1\n\t"
                              : "add t0, t0, t1\n\t";
  } else {
    asmStr += llvm::formatv((numTilesM > 1)
                                ? "addi t0, t0, {0}\n\taddi t5, t5, {0}\n\t"
                                : "addi t0, t0, {0}\n\t",
                            inBytes);
  }
  asmStr +=
      llvm::formatv("add t2, t2, t3\n\t{0}addi t6, t6, -1\n\tbnez t6, 1b\n\t",
                    numTilesN > 1 ? "add t4, t4, t3\n\t" : "");

  asmStr += emitTileWritebackAsm(numTilesM, numTilesN, "$4", "$5");
  createInlineAsm(b, loc, asmStr,
                  "{a0},{a1},{a2},{a3},{a4},{a5},~{t0},~{t1},~{t2},~{t3},"
                  "~{t4},~{t5},~{t6},~{v0},~{v4},~{v8},~{v12},~{v24},~{memory}",
                  {ptrA, strideA, ptrB, strideB, ptrC, strideC});
}

void lowerContractionToAsm(OpBuilder& b, vector::ContractionOp op,
                           int64_t numTilesM, int64_t numTilesN, Type inputElTy,
                           bool zeroTiles) {
  Location loc = op.getLoc();
  b.setInsertionPoint(op);
  SmallVector<Value> operands;
  for (int64_t m = 0; m < numTilesM; ++m)
    operands.push_back(
        castToScalable(b, loc, extractSubVector(b, loc, op.getLhs(), m)));
  for (int64_t n = 0; n < numTilesN; ++n)
    operands.push_back(
        castToScalable(b, loc, extractSubVector(b, loc, op.getRhs(), n)));

  std::string constraints;
  for (size_t i = 0; i < operands.size(); ++i)
    constraints += (i == 0 ? "^vr" : ",^vr");
  constraints += ",~{t5},~{t6}";

  bool isI8 = inputElTy.isInteger(8);
  // The vtzero block already programs the f32 config; emit a second one only
  // if that block was skipped, or if i8 needs the narrow config.
  std::string asmStr;
  if (zeroTiles) {
    asmStr = emitTileConfigAsm(/*isI8=*/false) +
             emitTileZeroAsm(numTilesM, numTilesN);
  }
  if (!zeroTiles || isI8) asmStr += emitTileConfigAsm(isI8);
  for (int64_t m = 0; m < numTilesM; ++m)
    for (int64_t n = 0; n < numTilesN; ++n)
      asmStr += llvm::formatv("{0} mt{1}, ${2}, ${3}\n\t",
                              isI8 ? "vtmms.tvv" : "vtfmm.tvv", (m * 2 + n) * 4,
                              m, numTilesM + n);
  createInlineAsm(b, loc, asmStr, constraints, operands);
  // The result lives in the mt tiles and is drained by the writeback asm;
  // replace the SSA value so it folds away.
  op.replaceAllUsesWith(
      arith::ConstantOp::create(b, loc, b.getZeroAttr(op.getType()))
          .getResult());
  op.erase();
}

std::pair<Value, SmallVector<Value>> extractBaseAndIndices(Value val) {
  val = traceToSourceValue(val);
  if (auto fe = val.getDefiningOp<vector::FromElementsOp>()) {
    for (Value op : fe.getOperands()) {
      if (auto te = op.getDefiningOp<vector::ToElementsOp>())
        op = te.getSource();
      if (auto ld = op.getDefiningOp<vector::LoadOp>())
        return {ld.getBase(), llvm::to_vector(ld.getIndices())};
    }
  }
  if (auto ld = val.getDefiningOp<vector::LoadOp>())
    return {ld.getBase(), llvm::to_vector(ld.getIndices())};
  if (auto rd = val.getDefiningOp<vector::TransferReadOp>())
    return {rd.getBase(), llvm::to_vector(rd.getIndices())};
  return {nullptr, {}};
}

bool isLhsTransposed(vector::ContractionOp op) {
  auto maps = op.getIndexingMapsArray();
  return maps.size() >= 3 && maps[0].getNumResults() >= 2 &&
         maps[2].getNumResults() == 2 &&
         maps[0].getResult(0) != maps[2].getResult(0);
}

std::tuple<Value, Value, Value> getBaseAndIndices(Operation* op) {
  if (auto w = dyn_cast<vector::TransferWriteOp>(op)) {
    auto indices = w.getIndices();
    if (indices.size() >= 2)
      return {w.getBase(), indices[indices.size() - 2], indices.back()};
  }
  return {nullptr, nullptr, nullptr};
}

SmallVector<AccumulatorChain> collectChains(mlir::FunctionOpInterface funcOp) {
  SmallVector<AccumulatorChain> chains;

  funcOp.walk([&](vector::ContractionOp contract) {
    if (!isSupportedMatrixContraction(contract)) return;

    auto yieldIt = llvm::find_if(contract->getUsers(), [](Operation* op) {
      return isa<scf::YieldOp>(op) &&
             isa_and_nonnull<scf::ForOp>(op->getParentOp());
    });
    if (yieldIt != contract->getUsers().end()) {
      auto forOp = cast<scf::ForOp>((*yieldIt)->getParentOp());

      SmallVector<Operation*> toErase;
      Value base = nullptr, row = nullptr, col = nullptr;

      for (Operation* u : forOp.getResult(0).getUsers()) {
        if (auto [b, r, c] = getBaseAndIndices(u); b) {
          base = b;
          row = r;
          col = c;
          toErase.push_back(u);
        } else if (auto ext = dyn_cast<vector::ExtractOp>(u)) {
          toErase.push_back(ext);
          for (Operation* eu : ext.getResult().getUsers()) {
            if (auto st = dyn_cast<vector::StoreOp>(eu)) {
              toErase.push_back(st);
              if (st.getIndices().size() >= 2 &&
                  (ext.getStaticPosition().empty() ||
                   ext.getStaticPosition().front() == 0)) {
                base = st.getBase();
                row = st.getIndices()[0];
                col = st.getIndices()[1];
              }
            }
          }
        }
      }
      if (base && !toErase.empty())
        chains.push_back({contract, base, row, col, forOp, std::move(toErase)});
    } else {
      for (Operation* u : contract.getResult().getUsers()) {
        if (auto [base, row, col] = getBaseAndIndices(u); base)
          chains.push_back({contract, base, row, col, nullptr, {u}});
      }
    }
  });
  return chains;
}

struct CoralNPUMatrixCodegenPass
    : public impl::CoralNPUMatrixCodegenBase<CoralNPUMatrixCodegenPass> {
  using CoralNPUMatrixCodegenBase::CoralNPUMatrixCodegenBase;

  void runOnOperation() override {
    auto funcOp = getOperation();
    if (!hasZvtTargetFeature(funcOp)) return;

    OpBuilder builder(&getContext());
    SmallVector<AccumulatorChain> chains = collectChains(funcOp);

    for (auto& chain : chains) {
      auto contractOp = chain.contractOp;
      auto accTy = cast<VectorType>(contractOp.getResult().getType());
      int64_t numTilesM = accTy.getDimSize(0) / 16;
      int64_t numTilesN = accTy.getDimSize(1) / 16;
      Type inputElTy = traceToSourceElementType(contractOp.getLhs());
      Location loc =
          chain.loop ? chain.loop.getLoc() : chain.opsToErase.front()->getLoc();
      auto eraseChainOps = [&]() {
        for (Operation* op : llvm::reverse(chain.opsToErase)) op->erase();
      };

      if (scf::ForOp forOp = chain.loop) {
        // Address computations share operands, so `hoisted` keeps the walk
        // linear. Later moves insert immediately before `forOp` and therefore
        // land after already-hoisted ops, preserving dominance.
        llvm::SmallPtrSet<Operation*, 8> hoisted;
        std::function<void(Operation*)> hoist = [&](Operation* op) {
          if (!op || (!forOp->isProperAncestor(op) &&
                      !(op->getBlock() == forOp->getBlock() &&
                        forOp->isBeforeInBlock(op)))) {
            return;
          }
          if (!hoisted.insert(op).second) return;
          for (Value operand : op->getOperands())
            hoist(operand.getDefiningOp());
          op->moveBefore(forOp);
        };
        for (Value v : {chain.baseC, chain.rowC, chain.colC}) {
          if (v) hoist(v.getDefiningOp());
        }

        auto [baseA, indicesA] = extractBaseAndIndices(contractOp.getLhs());
        auto [baseB, indicesB] = extractBaseAndIndices(contractOp.getRhs());

        std::optional<int64_t> lb = getConstantIntValue(forOp.getLowerBound());
        std::optional<int64_t> ub = getConstantIntValue(forOp.getUpperBound());
        std::optional<int64_t> step = getConstantIntValue(forOp.getStep());
        int64_t kTripCount =
            (lb && ub && step && *step > 0 && (*ub - *lb) % *step == 0)
                ? (*ub - *lb) / *step
                : 0;

        auto isValidFusedInit = [&](Value initVal) {
          Value src = traceToSourceValue(initVal);
          if (matchPattern(src, m_Zero()) ||
              matchPattern(src, m_AnyZeroFloat()))
            return true;
          auto [initBase, _] = extractBaseAndIndices(initVal);
          return initBase &&
                 getRootBuffer(initBase) == getRootBuffer(chain.baseC);
        };

        bool canUseFusedGEMM = baseA && baseB && kTripCount > 0 &&
                               forOp.getNumResults() == 1 &&
                               isValidFusedInit(forOp.getInitArgs()[0]);

        if (canUseFusedGEMM) {
          builder.setInsertionPoint(forOp);
          bool isTransposedA =
              isLhsTransposed(contractOp) ||
              (!indicesA.empty() && indicesA.back() != forOp.getInductionVar());
          Value c0 = arith::ConstantIndexOp::create(builder, loc, 0);
          auto getIdx = [](ArrayRef<Value> idxs, int64_t pos, Value def) {
            return (idxs.size() > pos &&
                    !isa<BlockArgument>(idxs[idxs.size() - 1 - pos]))
                       ? idxs[idxs.size() - 1 - pos]
                       : def;
          };
          Value rowA = isTransposedA ? c0 : getIdx(indicesA, 1, chain.rowC);
          Value colA = isTransposedA ? getIdx(indicesA, 0, chain.rowC) : c0;
          Value colB = getIdx(indicesB, 0, chain.colC);
          for (Value v : {baseA, baseB, rowA, colA, colB}) {
            if (v) hoist(v.getDefiningOp());
          }
          emitInlineFusedGEMM(builder, loc, baseA, rowA, colA, baseB, c0, colB,
                              chain.baseC, chain.rowC, chain.colC, numTilesM,
                              numTilesN, kTripCount, inputElTy, isTransposedA);
          eraseChainOps();
          forOp.erase();
          continue;
        }
      }
      // A K-reduction loop re-enters the multiply asm each iteration, so clear
      // the tiles once before it; a standalone contraction clears them inline.
      if (chain.loop) {
        builder.setInsertionPoint(chain.loop);
        createInlineAsm(builder, loc,
                        emitTileConfigAsm(/*isI8=*/false) +
                            emitTileZeroAsm(numTilesM, numTilesN),
                        "~{t5},~{t6}", {});
      }
      lowerContractionToAsm(builder, contractOp, numTilesM, numTilesN,
                            inputElTy, /*zeroTiles=*/!chain.loop);
      if (chain.loop)
        builder.setInsertionPointAfter(chain.loop);
      else
        builder.setInsertionPoint(chain.opsToErase.front());
      auto [ptrC, strideC] = extractPtrAndStride(
          builder, loc, chain.baseC, chain.rowC, chain.colC, kAccElemBytes);
      createInlineAsm(builder, loc,
                      emitTileWritebackAsm(numTilesM, numTilesN, "$0", "$1"),
                      "{a0},{a1},~{t0},~{t1},~{t4},~{t5},~{t6},~{v0},~{memory}",
                      {ptrC, strideC});
      eraseChainOps();
    }

    // Lower the contractions we did not take: LLVMCPUVirtualVectorLowering
    // skips contract lowering function-wide once it sees a Zvt-shaped one.
    bool hasContract =
        funcOp
            .walk([](vector::ContractionOp) { return WalkResult::interrupt(); })
            .wasInterrupted();
    if (!hasContract) return;

    RewritePatternSet patterns(&getContext());
    vector::populateVectorToVectorCanonicalizationPatterns(patterns);
    vector::populateVectorContractLoweringPatterns(
        patterns, vector::VectorContractLowering::OuterProduct);
    vector::populateScalarVectorTransferLoweringPatterns(
        patterns, /*benefit=*/1, /*allowMultipleUses=*/true);
    vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns))))
      signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createCoralNPUMatrixCodegenPass() {
  return std::make_unique<CoralNPUMatrixCodegenPass>();
}

}  // namespace mlir::coralnpu_compiler
