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

#include <functional>

#include "compiler/Transforms/CoralNPUTileSizeSelectionUtils.h"
#include "compiler/Transforms/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"

//===----------------------------------------------------------------------===//
// CoralNPU Matrix Codegen Pass
//
// Lowers vector.contract to Zvt matrix instructions (msetmtype, vtzero,
// vtmms.tvv / vtfmm.tvv / vtfmm.alt.tvv, vtmv).
//
// Supports FP32->FP32, BF16->FP32 and INT8->INT32 in 16x16, 16x32, 32x16,
// 32x32, 16x64, 64x16 and partial tile shapes. A contraction yielded by an
// scf.for reduction (constant bounds, step 1, k the row or innermost index of
// A and B, no memory writes in the body) replaces the whole loop with one
// inline-asm K loop holding the accumulators in mt0..mt12, also inside
// enclosing reduction loops (seeded before and drained after them); otherwise
// each iteration runs its own multiply block. +zvfbfwma BF16 vecmat K loops are
// lowered to vfwmaccbf16.vf.
//
// The tiles start from vtzero for a zero `acc`, or are seeded row by row
// (vle32.v + vtmv.t.v) when `acc` is a row-major tile load; any other `acc`
// is left to the generic lowering.
//===----------------------------------------------------------------------===//

namespace mlir::coralnpu_compiler {

#define GEN_PASS_DEF_CORALNPUMATRIXCODEGEN
#include "compiler/Transforms/Passes.h.inc"

namespace {

// A memref and the indices of a tile's first element.
struct TileAddr {
  Value base;
  SmallVector<Value> indices;
  int64_t rowStep = 1;
  explicit operator bool() const { return base != nullptr; }
};

// If `elems` are extracted from one vector at the flattened positions 0, step,
// 2 * step, ..., returns the vector and the step (1 for a single element).
std::pair<Value, int64_t> getStridedExtractSource(ValueRange elems) {
  auto ex0 = elems.front().getDefiningOp<vector::ExtractOp>();
  if (!ex0 || ex0.getSourceVectorType().isScalable()) return {};
  SmallVector<int64_t> basis =
      computeStrides(ex0.getSource().getType().getShape());
  int64_t step = 1;
  for (auto [i, e] : llvm::enumerate(elems)) {
    auto ex = e.getDefiningOp<vector::ExtractOp>();
    if (!ex || ex.getSource() != ex0.getSource() || ex.hasDynamicPosition() ||
        ex.getStaticPosition().size() != basis.size())
      return {};
    int64_t pos = linearize(ex.getStaticPosition(), basis);
    if (i == 1) step = pos;
    if (pos != static_cast<int64_t>(i) * step) return {};
  }
  return {ex0.getSource(), step};
}

// Peels widening casts and shape/broadcast wrappers to find the underlying
// source vector.
Value traceToSourceValue(Value val) {
  while (Operation* def = val.getDefiningOp()) {
    if (auto fe = dyn_cast<vector::FromElementsOp>(def)) {
      // In-order extracts from position 0 stand for their source vector.
      auto [src, step] = getStridedExtractSource(fe.getElements());
      if (!src || step != 1 ||
          cast<VectorType>(src.getType()).getNumElements() <= 1)
        break;
      val = src;
      continue;
    }
    // Only a zero-offset, unit-stride slice starts where its source starts.
    if (auto slice = dyn_cast<vector::ExtractStridedSliceOp>(def)) {
      if (slice.hasNonUnitStrides() ||
          !llvm::all_of(slice.getOffsets(), isZeroInteger))
        break;
    } else if (!isa<arith::ExtFOp, arith::ExtSIOp, vector::BroadcastOp,
                    vector::ShapeCastOp>(def)) {
      break;
    }
    val = def->getOperand(0);
  }
  return val;
}

// Drops a widening ext or shape_cast, as the asm reads the narrow input
// elements.
Value stripExt(Value v) {
  while (auto sc = v.getDefiningOp<vector::ShapeCastOp>()) v = sc.getSource();
  Operation* d = v.getDefiningOp();
  return isa_and_nonnull<arith::ExtFOp, arith::ExtSIOp>(d) ? d->getOperand(0)
                                                           : v;
}

bool isSupportedMatrixContraction(vector::ContractionOp op, bool& swap) {
  auto accTy = dyn_cast<VectorType>(op.getAccType());
  if (!accTy || (accTy.getRank() != 1 && accTy.getRank() != 2) ||
      accTy.isScalable())
    return false;
  SmallVector<AffineMap> maps = op.getIndexingMapsArray();
  bool rank1 = accTy.getRank() == 1;
  unsigned nDim = maps[2].getDimPosition(rank1 ? 0 : 1);
  unsigned mDim = rank1 ? nDim : maps[2].getDimPosition(0);
  swap = !rank1 && maps[1].isFunctionOfDim(mDim);
  if (swap) std::swap(maps[0], maps[1]);
  Value lhs = swap ? op.getRhs() : op.getLhs();
  Value rhs = swap ? op.getLhs() : op.getRhs();
  int64_t m = rank1 ? 1 : accTy.getDimSize(0), n = accTy.getShape().back();
  if ((m % 16 != 0 && (m > 16 || swap)) || m == 48 || n == 48 ||
      (n % 16 != 0 && ((n != 8 && n != 12) || rank1 || swap)) ||
      llvm::alignTo(m, 16) * llvm::alignTo(n, 16) > 1024 ||
      cast<VectorType>(lhs.getType()).getNumElements() != m ||
      cast<VectorType>(rhs.getType()).getNumElements() != n ||
      !isZvtElementTypes(getElementTypeOrSelf(traceToSourceValue(lhs)),
                         getElementTypeOrSelf(traceToSourceValue(rhs)),
                         accTy.getElementType()))
    return false;
  auto iters = op.getIteratorTypes().getValue();
  return op.getKind() == vector::CombiningKind::ADD &&
         vector::isParallelIterator(iters[mDim]) &&
         vector::isParallelIterator(iters[nDim]) &&
         maps[0].isFunctionOfDim(mDim) != rank1 &&
         maps[1].isFunctionOfDim(nDim);
}

void createInlineAsm(OpBuilder& b, Location loc, StringRef asmStr,
                     StringRef constraints, ValueRange operands) {
  LLVM::InlineAsmOp::create(
      b, loc, TypeRange{}, operands, b.getStringAttr(asmStr),
      b.getStringAttr(constraints), /*has_side_effects=*/b.getUnitAttr(),
      /*is_align_stack=*/nullptr, /*tail_call_kind=*/nullptr,
      /*convergent=*/nullptr, /*asm_dialect=*/nullptr,
      /*operand_attrs=*/nullptr);
}

// Returns the byte address of `addr` and its row stride in bytes.
// The innermost stride must be 1.
std::array<Value, 2> extractPtrAndStride(OpBuilder& b, Location loc,
                                         const TileAddr& addr) {
  auto s = memref::ExtractStridedMetadataOp::create(b, loc, addr.base);
  auto strides = s.getStrides();
  Value base =
      memref::ExtractAlignedPointerAsIndexOp::create(b, loc, s.getBaseBuffer());
  int64_t elBytes = getElementTypeOrSelf(addr.base).getIntOrFloatBitWidth() / 8;
  Value bytes = arith::ConstantIndexOp::create(b, loc, elBytes);
  Value off = arith::AddIOp::create(b, loc, s.getOffset(), addr.indices.back());
  for (auto [idx, stride] : llvm::zip(llvm::drop_end(addr.indices), strides))
    off = arith::AddIOp::create(b, loc, off,
                                arith::MulIOp::create(b, loc, idx, stride));
  Value rowStride =
      arith::ConstantIndexOp::create(b, loc, elBytes * addr.rowStep);
  if (strides.size() >= 2)
    rowStride = arith::MulIOp::create(b, loc, strides.end()[-2], rowStride);
  Value ptr = arith::AddIOp::create(b, loc, base,
                                    arith::MulIOp::create(b, loc, off, bytes));
  return {arith::IndexCastOp::create(b, loc, b.getI32Type(), ptr),
          arith::IndexCastOp::create(b, loc, b.getI32Type(), rowStride)};
}

// A clobber names one register, so list each LMUL4 group in full.
constexpr const char* kClobbers =
    "~{t0},~{t1},~{t2},~{t3},~{t4},~{t5},~{t6},~{v0},~{v1},~{v2},~{v3},"
    "~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},"
    "~{v14},~{v15},~{v16},~{v17},~{v18},~{v19},~{v20},~{v21},~{v22},~{v23},"
    "~{v24},~{v25},~{v26},~{v27},~{v28},~{v29},~{v30},~{v31},~{memory}";

// msetmtype operands (decode: third_party/coralnpu .../design/RvvFrontEnd.sv).
// mtype (rs1): tm = bits[23:10], tk = bits[7:5], mtwiden = bits[1:0]. Always
// TM=16; mtwiden is 3 for i8 (TWIDEN=4), 2 for bf16 and 1 for f32.
// vtype (rs2): SEW = bits[5:3], LMUL = bits[2:0], altfmt = bit 8. altfmt makes
// the *second* operand signed INT8 / BF16; the opcode does so for the first
// (vtmms, vtfmm.alt). FP32 leaves it clear. vtype is 256 (SEW8, LMUL1, altfmt)
// for i8, 265 (SEW16, LMUL2, altfmt) for bf16 and 18 (SEW32, LMUL4) for f32.
// Programs mtype/vtype, then TN (msetmtype zeroes vl). Emitted per block, as
// LLVM may insert a vset* between blocks, which clears mtype and vtype.altfmt.
// Clobbers t5 and t6. A null `elTy` selects the 32-bit accumulator config.
std::string emitTileConfigAsm(Type elTy = {}, int64_t tk = 1, int64_t tn = 16) {
  int w = elTy && elTy.isInteger(8) ? 3 : (elTy && elTy.isBF16() ? 2 : 1);
  return llvm::formatv(
      "li t5, {0}\n\tli t6, {1}\n\tmsetmtype t5, t6\n\tli t6, {2}\n\t"
      "msettn zero, t6\n\t",
      16384 | (tk << 5) | w, w == 3 ? 256 : (w == 2 ? 265 : 18), tn);
}

StringRef getMatmulOp(Type elTy) {
  if (elTy.isInteger(8)) return "vtmms.tvv";
  return elTy.isBF16() ? "vtfmm.alt.tvv" : "vtfmm.tvv";
}

// Drains accumulator tiles mt0..mt12 to memory via vtmv.v.t + vse32.v (for a
// bf16 `outElTy`, vfncvtbf16.f.f.w + vse16.v, 4 rows at a time) or, with a null
// `outElTy`, seeds them from memory via vle32.v + vtmv.t.v.
// `ptrReg`/`strideReg` name the asm operands holding the pointer and stride.
// Clobbers t0-t6 and v0-v31.
std::string emitTileRowsAsm(int64_t numTilesM, int64_t numTilesN,
                            StringRef ptrReg, StringRef strideReg, Type outElTy,
                            int64_t rowsM, int64_t colsN) {
  bool isBF16 = outElTy && outElTy.isBF16();
  int64_t tn = std::min<int64_t>(colsN, 16);
  std::string asmStr = (isBF16 ? llvm::formatv("li t1, {0}\n\t", tn).str()
                               : emitTileConfigAsm({}, 1, tn)) +
                       llvm::formatv("mv t5, {0}\n\t", strideReg).str();
  const char* lui[] = {"li t0, 0\n\t", "lui t0, 0x20000\n\t",
                       "lui t0, 0x40000\n\t", "lui t0, 0x60000\n\t"};
  std::string move;
  if (isBF16) {
    // Move 4 rows under the f32 config (TK=1), then narrow them under the bf16
    // one.
    move =
        "li t2, 16417\n\tli t3, 18\n\tmsetmtype t2, t3\n\tmsettn zero, t1\n\t";
    for (int r : {8, 16, 20, 24})
      move += llvm::formatv("vtmv.v.t v{0}, t0\n\taddi t0, t0, 1\n\t", r);
    move +=
        "li t2, 16418\n\tli t3, 265\n\tmsetmtype t2, t3\n\tmsettn zero, t1\n\t";
    for (int r : {8, 16, 20, 24})
      move += llvm::formatv(
          "vfncvtbf16.f.f.w v12, v{0}\n\tvse16.v v12, (t4)\n\t"
          "add t4, t4, t5\n\t",
          r);
  } else {
    move = std::string(!outElTy ? "vle32.v v0, (t4)\n\tvtmv.t.v t0, v0\n\t"
                                : "vtmv.v.t v0, t0\n\tvse32.v v0, (t4)\n\t") +
           "add t4, t4, t5\n\taddi t0, t0, 1\n\t";
  }
  int64_t rows = std::min<int64_t>(rowsM, 16);
  int64_t iters = isBF16 ? rows / 4 : rows;
  int64_t unroll = !isBF16 && iters % 4 == 0 ? 4 : 1;
  std::string body;
  for (int64_t u = 0; u < unroll; ++u) body += move;
  for (int64_t m = 0; m < numTilesM; ++m) {
    for (int64_t n = 0; n < numTilesN; ++n) {
      int64_t t = m * numTilesN + n;
      if (!m || numTilesN > 1)
        asmStr +=
            m ? llvm::formatv("slli t4, t5, 4\n\tadd t4, {0}, t4\n\t", ptrReg)
              : llvm::formatv("mv t4, {0}\n\t", ptrReg);
      if (n)
        asmStr += llvm::formatv("addi t4, t4, {0}\n\t", n * (isBF16 ? 32 : 64));
      if (iters == unroll)
        asmStr += llvm::formatv("{0}{1}", lui[t], body);
      else
        asmStr += llvm::formatv(
            "{0}li t6, {1}\n\t{2}:\n\t{3}addi t6, t6, -1\n\tbnez t6, {2}b\n\t",
            lui[t], iters / unroll, t + 2, body);
    }
  }
  return asmStr;
}

// Clears the tiles (vtfmm/vtmms accumulate in place) under the 32-bit
// accumulator config. Clobbers t5 and t6.
std::string emitTileZeroAsm(int64_t numTilesM, int64_t numTilesN) {
  std::string asmStr = emitTileConfigAsm();
  for (int64_t t = 0; t < numTilesM * numTilesN; ++t)
    asmStr += llvm::formatv("vtzero mt{0}\n\t", t * 4);
  return asmStr;
}

// Returns the base register of Zvt operand slot `slot`. Zvt v0.4 operand bases
// are LMUL-aligned (LMUL = inBytes) and in the first 8 / KMAX registers of
// their 8-register group (KMAX = 4 / inBytes), so slot pairs share a group.
int64_t getZvtOperandBase(int64_t slot, int64_t inBytes) {
  return 8 * (slot / 2) + (slot % 2) * inBytes;
}

// `contigA`: k indexes the rows of A, so each K step loads a contiguous row;
// otherwise a strided column. B is analogous (`contigB`).
void emitInlineFusedGEMM(OpBuilder& b, Location loc, const TileAddr& lhs,
                         const TileAddr& rhs, const TileAddr& dst,
                         const TileAddr& init, int64_t numTilesM,
                         int64_t numTilesN, int64_t numKSteps, Type inputElTy,
                         bool contigA, bool contigB, int64_t rowsM,
                         int64_t colsN, int64_t tk, bool configSet = false) {
  int64_t inBytes = inputElTy.getIntOrFloatBitWidth() / 8;
  int64_t step = numKSteps >= 32 && numKSteps % (8 * tk) == 0   ? 8 * tk
                 : numKSteps >= 16 && numKSteps % (4 * tk) == 0 ? 4 * tk
                 : numKSteps <= 4 * tk                          ? numKSteps
                                                                : tk;
  SmallVector<Value> operands;
  for (const TileAddr& a : {lhs, rhs, dst, init})
    if (a) llvm::append_range(operands, extractPtrAndStride(b, loc, a));
  // The tile init and narrow config precede the pointer setup: they use t0 and
  // t4-t6 as scratch, which the K loop then holds.
  std::string asmStr;
  if (dst)
    asmStr = init ? emitTileRowsAsm(numTilesM, numTilesN, "$6", "$7", {}, rowsM,
                                    colsN)
                  : emitTileZeroAsm(numTilesM, numTilesN);
  if (!configSet && (!dst || !inputElTy.isF32() || (init && colsN < 16)))
    asmStr += emitTileConfigAsm(inputElTy, tk);
  asmStr += "mv t0, $0\n\tmv t1, $1\n\tmv t2, $2\n\tmv t3, $3\n\t";

  // A: pointer t0 (second M tile t5), stride t1. B: t2 (second N tile t4), t3.
  // A 4-tile strip walks its second-tile register and, if strided, keeps the
  // 16-row step in the other one.
  struct Opnd {
    bool contig;
    int64_t tiles;
    const char* p[3];
    int64_t slots[4];
    const char* s;
  } ab[] = {{contigA,
             numTilesM,
             {"t0", "t5", "t4"},
             {contigA && numTilesM > 1 ? 6 : 1,
              contigA && numTilesM > 1 ? 7 : 6, 4, 5},
             "t1"},
            {contigB, numTilesN, {"t2", "t4", "t5"}, {2, 3, 4, 5}, "t3"}};
  auto vReg = [&](int64_t s, int64_t kk) {
    return getZvtOperandBase(s, inBytes) + kk * (8 / tk);
  };
  auto tStep = [&](const Opnd& o) {
    return o.contig && tk == 4 / inBytes && o.tiles > 1 ? 2 : 1;
  };
  auto fmt = [&](const char* f, auto&&... args) {
    asmStr += llvm::formatv(f, args...).str();
  };
  for (Opnd& o : ab)
    if (o.tiles == 2 && tStep(o) == 1)
      fmt(o.contig ? "addi {0}, {1}, {2}\n\t"
                   : "slli {0}, {2}, 4\n\tadd {0}, {1}, {0}\n\t",
          o.p[1], o.p[0], o.contig ? std::to_string(16 * inBytes) : o.s);
    else if (o.tiles > 2 && !o.contig)
      fmt("slli {0}, {1}, 4\n\t", o.p[2], o.s);
  // A partial tile's operand loads still take 16 elements, so mask them to the
  // rowsM (A) or colsN (B) valid ones to stay within the operand. With both
  // partial, the masks wait in v0-v7 registers clear of slot 1's A operand
  // (v1/v3/v5/v7 for i8, v2-v3/v6-v7 for bf16, v4-v7 for f32) and are copied
  // to v0 before each operand's loads.
  bool masked[2] = {rowsM < 16, colsN < 16};
  int64_t maskReg[2] = {masked[0] && masked[1] ? (inBytes == 1 ? 2 : 1) : 0,
                        masked[0] && masked[1] ? (inBytes == 4 ? 2 : 4) : 0};
  if (masked[0] || masked[1]) fmt("vid.v v8\n\t");
  for (int i = 0; i < 2; ++i)
    if (masked[i])
      fmt("li t6, {0}\n\tvmsltu.vx v{1}, v8, t6\n\t", i ? colsN : rowsM,
          maskReg[i]);
  if (numKSteps > step) fmt("li t6, {0}\n\t1:\n\t", numKSteps / step);
  for (int64_t s = 0; s < step; ++s) {
    int64_t kk = s % tk;
    for (auto [i, o] : llvm::enumerate(ab)) {
      if (masked[0] && masked[1]) fmt("vmv1r.v v0, v{0}\n\t", maskReg[i]);
      for (int64_t t = 0; t < o.tiles; t += tStep(o)) {
        if (o.tiles > 2 && t > 0)
          fmt("add{0} {1}, {2}, {3}\n\t", o.contig ? "i" : "", o.p[1],
              o.contig || t == 1 ? o.p[0] : o.p[1],
              o.contig ? std::to_string(16 * inBytes * t) : o.p[2]);
        fmt("vl{0}e{1}.v v{2}, ({3}){4}{5}\n\t",
            tStep(o) == 2 ? std::to_string(2 * inBytes) + "r"
                          : (o.contig ? "" : "s"),
            inBytes * 8, vReg(o.slots[t], kk), o.p[std::min<int64_t>(t, 1)],
            o.contig ? "" : std::string(", ") + o.s, masked[i] ? ", v0.t" : "");
      }
    }
    if (kk + 1 == tk)
      for (int64_t m = 0; m < numTilesM; ++m)
        for (int64_t n = 0; n < numTilesN; ++n)
          fmt("{0} mt{1}, v{2}, v{3}\n\t", getMatmulOp(inputElTy),
              (m * numTilesN + n) * 4, vReg(ab[0].slots[m], 0),
              vReg(ab[1].slots[n], 0));
    if (s + 1 < numKSteps)
      for (Opnd& o : ab)
        for (int64_t t = 0; t < (o.tiles == 2 && tStep(o) == 1 ? 2 : 1); ++t)
          fmt("add{0} {1}, {1}, {2}\n\t", o.contig ? "" : "i", o.p[t],
              o.contig ? o.s : std::to_string(inBytes));
  }
  if (numKSteps > step) asmStr += "addi t6, t6, -1\n\tbnez t6, 1b\n\t";
  if (dst)
    asmStr += emitTileRowsAsm(numTilesM, numTilesN, "$4", "$5",
                              getElementTypeOrSelf(dst.base), rowsM, colsN);
  std::string constraints = std::string("{a0},{a1},{a2},{a3},") +
                            (dst ? "{a4},{a5}," : "") +
                            (init ? "{a6},{a7}," : "") + kClobbers;
  createInlineAsm(b, loc, asmStr, constraints, operands);
}

void lowerContractionToAsm(OpBuilder& b, vector::ContractionOp op,
                           int64_t numTilesM, int64_t numTilesN, Type inputElTy,
                           bool zeroTiles) {
  Location loc = op.getLoc();
  b.setInsertionPoint(op);
  SmallVector<Value> operands;
  std::string constraints;
  Value zero =
      LLVM::ConstantOp::create(b, loc, b.getI64Type(), b.getI64IntegerAttr(0));
  int64_t slot = 0;
  for (auto [v, tiles] : {std::pair{stripExt(op.getLhs()), numTilesM},
                          std::pair{stripExt(op.getRhs()), numTilesN}}) {
    auto ty = cast<VectorType>(v.getType());
    Type elTy = ty.getElementType();
    int64_t inBytes = elTy.getIntOrFloatBitWidth() / 8;
    auto matTy = VectorType::get({ty.getNumElements() / 16, 16}, elTy);
    // One 16-element row (LMUL = inBytes at VLEN 128), pinned to the operand
    // slot bases.
    VectorType scTy = VectorType::get({8}, elTy, {true});
    for (int64_t t = 0; t < tiles; ++t, ++slot) {
      Value flat = vector::ExtractOp::create(
          b, loc, vector::ShapeCastOp::create(b, loc, matTy, v), t);
      operands.push_back(
          LLVM::CallIntrinsicOp::create(
              b, loc, scTy, b.getStringAttr("llvm.vector.insert"),
              ValueRange{LLVM::PoisonOp::create(b, loc, scTy), flat, zero})
              .getResult(0));
      constraints +=
          llvm::formatv("{{v{0}},", getZvtOperandBase(slot, inBytes)).str();
    }
  }
  constraints += "~{t5},~{t6}";
  // Program the config unless the vtzero block left the right (f32) one.
  std::string asmStr = zeroTiles ? emitTileZeroAsm(numTilesM, numTilesN) : "";
  if (!zeroTiles || !inputElTy.isF32()) asmStr += emitTileConfigAsm(inputElTy);
  for (int64_t m = 0; m < numTilesM; ++m)
    for (int64_t n = 0; n < numTilesN; ++n)
      asmStr +=
          llvm::formatv("{0} mt{1}, ${2}, ${3}\n\t", getMatmulOp(inputElTy),
                        (m * numTilesN + n) * 4, m, numTilesM + n);
  createInlineAsm(b, loc, asmStr, constraints, operands);
  // The drain asm writes the result from the tiles; zero is a placeholder.
  op.replaceAllUsesWith(
      arith::ConstantOp::create(b, loc, b.getZeroAttr(op.getType()))
          .getResult());
  op.erase();
}

// Returns the address of a row-major access: rank >= minRank with unit
// innermost stride, else null. A subview that drops a trailing unit dim (e.g.
// a 1-channel conv image) is addressed in its source, as an N x 1 tile.
TileAddr getRowMajorAddr(Value base, ValueRange indices, size_t minRank = 2) {
  auto ty = dyn_cast<MemRefType>(base.getType());
  if (!ty || indices.size() < minRank || !ty.isLastDimUnitStride()) return {};
  SmallVector<Value> idx(indices);
  if (auto sv = base.getDefiningOp<memref::SubViewOp>()) {
    MemRefType srcTy = sv.getSourceType();
    llvm::SmallBitVector dropped = sv.getDroppedDims();
    if (srcTy.getShape().back() == 1 && dropped.count() == 1 &&
        dropped.test(srcTy.getRank() - 1) && sv.hasZeroOffset() &&
        sv.hasUnitStride()) {
      // Index the dropped dim with an entry-block 0, which dominates all uses.
      Block& entry =
          sv->getParentOfType<FunctionOpInterface>().getFunctionBody().front();
      auto consts = entry.getOps<arith::ConstantIndexOp>();
      auto it = llvm::find_if(consts, [](auto c) { return c.value() == 0; });
      OpBuilder b = OpBuilder::atBlockBegin(&entry);
      Value zero = it != consts.end()
                       ? Value(*it)
                       : arith::ConstantIndexOp::create(b, sv.getLoc(), 0);
      base = sv.getSource();
      ty = srcTy;
      idx.push_back(zero);
    }
  }
  SmallVector<int64_t> strides;
  int64_t offset;
  if (succeeded(ty.getStridesAndOffset(strides, offset)) && idx.size() >= 3 &&
      strides.end()[-2] > 0 && strides.end()[-3] == strides.end()[-2] &&
      isConstantIntValue(idx.end()[-2], 0))
    std::swap(idx.end()[-2], idx.end()[-3]);
  return {base, std::move(idx)};
}

// A transfer the asm can replace: unmasked, row-major and, if `inBounds`, in
// bounds.
TileAddr getTransferAddr(Operation* op, bool inBounds = true,
                         size_t minRank = 2) {
  auto xfer = dyn_cast_or_null<VectorTransferOpInterface>(op);
  if (!xfer || xfer.getMask() || (inBounds && xfer.hasOutOfBoundsDim()) ||
      !xfer.getPermutationMap().isMinorIdentity())
    return {};
  return getRowMajorAddr(xfer.getBase(), xfer.getIndices(), minRank);
}

// Returns `v1 - v0` if it is a known constant.
std::optional<int64_t> getIndexDiff(Value v1, Value v0) {
  if (v1 == v0) return 0;
  if (auto c0 = getConstantIntValue(v0), c1 = getConstantIntValue(v1); c0 && c1)
    return *c1 - *c0;
  if (Operation *d1 = v1.getDefiningOp(), *d0 = v0.getDefiningOp();
      d1 && d0 && d1->getName() == d0->getName() && d1->getNumResults() == 1 &&
      isMemoryEffectFree(d1) && d1->getAttrs() == d0->getAttrs() &&
      llvm::equal(d1->getOperands(), d0->getOperands(),
                  [](Value x, Value y) { return getIndexDiff(x, y) == 0; }))
    return 0;
  if (auto add = v1.getDefiningOp<arith::AddIOp>())
    for (int i : {0, 1}) {
      if (getIndexDiff(add->getOperand(i), v0) == 0)
        return getConstantIntValue(add->getOperand(1 - i));
      if (auto a0 = v0.getDefiningOp<arith::AddIOp>();
          a0 && getIndexDiff(add->getOperand(i), a0->getOperand(i)) == 0)
        return getIndexDiff(add->getOperand(1 - i), a0->getOperand(1 - i));
    }
  return std::nullopt;
}

// True if `a` addresses row `r` of the tile at `a0`.
bool isTileRow(const TileAddr& a, const TileAddr& a0, int64_t r) {
  auto eq = [](Value x, Value y) { return getIndexDiff(x, y) == 0; };
  ArrayRef<Value> idx = a.indices, idx0 = a0.indices;
  return a.base == a0.base && idx.size() >= 2 && eq(idx.back(), idx0.back()) &&
         llvm::equal(idx.drop_back(2), idx0.drop_back(2), eq) &&
         getIndexDiff(idx.end()[-2], idx0.end()[-2]) == r;
}

// Returns the address of a row-major tile load, or null.
TileAddr getLoadAddr(Value v, bool trace = false, size_t minRank = 2) {
  if (trace) v = traceToSourceValue(v);
  if (auto fe = v.getDefiningOp<vector::FromElementsOp>()) {
    ValueRange elems = fe.getElements();
    // Elements of an N x 1 column tile (see getRowMajorAddr) extracted at a
    // constant step (e.g. strided 1-channel conv) are rows at that step.
    VectorType feTy = fe.getType();
    if (auto [src, step] = getStridedExtractSource(elems);
        src && step > 0 && feTy.getRank() > 0 &&
        feTy.getNumElements() == feTy.getDimSize(0)) {
      TileAddr a = getLoadAddr(src, /*trace=*/false, minRank);
      if (a && cast<MemRefType>(a.base.getType()).getShape().back() == 1) {
        a.rowStep *= step;
        return a;
      }
    }
    auto te0 = elems.front().getDefiningOp<vector::ToElementsOp>();
    size_t cols = te0 ? te0.getNumResults() : 1, rows = elems.size() / cols;
    TileAddr row0;
    for (size_t r = 0; r < rows; ++r) {
      Value e = elems[r * cols];
      TileAddr a;
      if (auto mld = e.getDefiningOp<memref::LoadOp>(); mld && !te0)
        a = getRowMajorAddr(mld.getMemRef(), mld.getIndices(), minRank);
      else if (auto ex = e.getDefiningOp<vector::ExtractOp>();
               ex && !te0 && ex.getSource().getType().getNumElements() == 1)
        a = getLoadAddr(ex.getSource(), trace, minRank);
      else if (auto te = e.getDefiningOp<vector::ToElementsOp>();
               te && elems.slice(r * cols, cols) == ValueRange(te.getResults()))
        a = getLoadAddr(te.getSource(), trace, minRank);
      if (r == 0)
        row0 = a;
      else if (r == 1 && !te0 && a.indices.size() >= 2)
        row0.rowStep = getIndexDiff(a.indices.end()[-2], row0.indices.end()[-2])
                           .value_or(0);
      if (row0.rowStep <= 0 || !isTileRow(a, row0, r * row0.rowStep)) return {};
    }
    return row0;
  }
  if (auto ld = v.getDefiningOp<vector::LoadOp>())
    return getRowMajorAddr(ld.getBase(), ld.getIndices(), minRank);
  return getTransferAddr(v.getDefiningOp(), !trace, minRank);
}

// When `a`'s row and column dims both have unit stride (e.g. 1-channel conv)
// and its column index is 0, splits a row index `iv + x` into row `iv` and
// column `x` (if `rowStep == 1`), else into row `x` and column `iv`.
void splitKIndex(TileAddr& a, Value iv) {
  SmallVector<int64_t> strides;
  int64_t offset;
  if (a.indices.size() < 2 || !isConstantIntValue(a.indices.back(), 0) ||
      failed(cast<MemRefType>(a.base.getType())
                 .getStridesAndOffset(strides, offset)) ||
      strides.end()[-2] != 1 || strides.back() != 1)
    return;
  Value &row = a.indices.end()[-2], &col = a.indices.back();
  Value rest = row == iv ? col : Value();
  if (auto add = row.getDefiningOp<arith::AddIOp>())
    rest = add.getLhs() == iv ? add.getRhs()
                              : (add.getRhs() == iv ? add.getLhs() : rest);
  if (!rest) return;
  row = a.rowStep == 1 ? iv : rest;
  col = a.rowStep == 1 ? rest : iv;
}

// Returns 0 if `iv` is the innermost index of a row-major access, 1 if it is
// the row index, else -1.
int64_t getKPos(const TileAddr& a, Value iv) {
  if (llvm::count(a.indices, iv) != 1) return -1;
  return a.indices.back() == iv
             ? 0
             : (a.rowStep == 1 && a.indices.end()[-2] == iv ? 1 : -1);
}

// Returns the tile whose writeback of `res` the drain replaces, and the ops to
// erase: either a lone store or transfer_write, or row extracts stored to one
// tile, optionally after a bf16 truncf (`allowTrunc`) or, with `transposedOut`,
// a [1, 0] transpose. The drain writes all rows and goes right after the loop,
// so the writes it replaces must be in `block`.
std::pair<TileAddr, SmallVector<Operation*>> matchTileStore(
    Value res, Block* block, bool allowTrunc, bool* transposedOut = nullptr) {
  SmallVector<Operation*> toErase, worklist;
  auto inBlock = [&](Operation* op) { return op->getBlock() == block; };
  if (auto trunc = dyn_cast_or_null<arith::TruncFOp>(
          allowTrunc && res.hasOneUse() ? *res.user_begin() : nullptr);
      trunc && !trunc.getRoundingmode() &&
      getElementTypeOrSelf(trunc.getType()).isBF16()) {
    toErase.push_back(trunc);
    res = trunc.getResult();
  }
  for (Operation* u : res.getUsers())
    if (!transposedOut || !OpTrait::hasElementwiseMappableTraits(u))
      worklist.push_back(u);
  auto vecTy = cast<VectorType>(res.getType());
  int64_t m = vecTy.getDimSize(0),
          n = vecTy.getRank() > 1 ? vecTy.getDimSize(1) : 0;
  bool canTranspose = transposedOut && vecTy.getElementType().isF32();
  if (auto tr = dyn_cast_or_null<vector::TransposeOp>(
          canTranspose && worklist.size() == 1 ? worklist[0] : nullptr);
      tr && tr.getPermutation() == ArrayRef<int64_t>{1, 0}) {
    toErase.push_back(tr);
    res = tr.getResult();
    *transposedOut = true;
    std::swap(m, n);
    worklist.assign(res.user_begin(), res.user_end());
  }
  if (worklist.size() == 1) {
    auto st = dyn_cast<vector::StoreOp>(worklist[0]);
    if (TileAddr c = st && !n
                         ? getRowMajorAddr(st.getBase(), st.getIndices(), 1)
                         : getTransferAddr(worklist[0])) {
      toErase.push_back(worklist[0]);
      if (llvm::all_of(toErase, inBlock))
        return {std::move(c), std::move(toErase)};
      return {};
    }
  }
  if (!n) return {};

  DenseMap<Value, SmallVector<int64_t>> elemMap;
  std::function<SmallVector<int64_t>(Value)> getElems =
      [&](Value v) -> SmallVector<int64_t> {
    if (auto it = elemMap.find(v); it != elemMap.end()) return it->second;
    auto vTy = dyn_cast<VectorType>(v.getType());
    Operation* def = v.getDefiningOp();
    if (!vTy || vTy.getRank() != 1 || !def) return {};
    SmallVector<int64_t> elems(vTy.getNumElements(), -1);
    if (def->getNumOperands() == 0) return elemMap[v] = std::move(elems);
    if (!inBlock(def)) return {};
    if (auto ext = dyn_cast<vector::ExtractOp>(def);
        ext && ext.getSource() == res && ext.getStaticPosition().size() == 1) {
      int64_t r = ext.getStaticPosition()[0];
      if (r < 0 || r >= m) return {};
      for (int64_t j = 0; j < n; ++j) elems[j] = r * n + j;
    } else if (auto ins = dyn_cast<vector::InsertStridedSliceOp>(def);
               ins && !ins.hasNonUnitStrides()) {
      auto src = getElems(ins.getValueToStore()), dst = getElems(ins.getDest());
      if (src.empty() || dst.empty()) return {};
      elems = std::move(dst);
      llvm::copy(
          src, elems.begin() + cast<IntegerAttr>(ins.getOffsets()[0]).getInt());
    } else if (auto shuf = dyn_cast<vector::ShuffleOp>(def)) {
      auto v1 = getElems(shuf.getV1()), v2 = getElems(shuf.getV2());
      if (v1.empty() || v2.empty()) return {};
      ArrayRef<int64_t> mask = shuf.getMask();
      for (int64_t i = 0, s1 = v1.size(); i < (int64_t)mask.size(); ++i)
        elems[i] =
            mask[i] < 0 ? -1 : (mask[i] < s1 ? v1[mask[i]] : v2[mask[i] - s1]);
    } else if (auto sl = dyn_cast<vector::ExtractStridedSliceOp>(def);
               sl && !sl.hasNonUnitStrides()) {
      auto src = getElems(sl.getSource());
      if (src.empty()) return {};
      int64_t off = cast<IntegerAttr>(sl.getOffsets()[0]).getInt();
      elems.assign(src.begin() + off, src.begin() + off + elems.size());
    } else {
      return {};
    }
    toErase.push_back(def);
    return elemMap[v] = std::move(elems);
  };

  DenseSet<Operation*> visited;
  SmallVector<vector::StoreOp> stores;
  while (!worklist.empty()) {
    Operation* u = worklist.pop_back_val();
    if (!visited.insert(u).second) continue;
    if (auto st = dyn_cast<vector::StoreOp>(u)) {
      if (!inBlock(st)) return {};
      stores.push_back(st);
      continue;
    }
    if (u->getNumResults() != 1 || getElems(u->getResult(0)).empty() ||
        u->use_empty())
      return {};
    worklist.append(u->user_begin(), u->user_end());
  }
  if (stores.empty()) return {};

  auto matchRow = [&](ArrayRef<int64_t> elems, bool tr) -> int64_t {
    int64_t cols = tr ? m : n;
    if ((int64_t)elems.size() != cols || elems[0] < 0) return -1;
    int64_t r = tr ? (elems[0] % n) : (elems[0] / n);
    for (int64_t j = 0; j < cols; ++j)
      if (elems[j] != (tr ? j * n + r : r * n + j)) return -1;
    return r;
  };
  auto elems0 = getElems(stores[0].getValueToStore());
  bool isTr = matchRow(elems0, false) < 0;
  if (isTr && (!canTranspose || matchRow(elems0, true) < 0)) return {};
  if (isTr) *transposedOut = true;
  SmallVector<TileAddr> rows(isTr ? n : m);
  for (vector::StoreOp st : stores) {
    int64_t r = matchRow(getElems(st.getValueToStore()), isTr);
    TileAddr a = getRowMajorAddr(st.getBase(), st.getIndices());
    if (r < 0 || !a || rows[r]) return {};
    toErase.push_back(st);
    rows[r] = std::move(a);
  }
  for (size_t r = 0; r < rows.size(); ++r)
    if (!isTileRow(rows[r], rows[0], r)) return {};
  llvm::sort(toErase,
             [](Operation* a, Operation* b) { return a->isBeforeInBlock(b); });
  return {std::move(rows[0]), std::move(toErase)};
}

bool hasContraction(Operation* op) {
  return op->walk([](vector::ContractionOp) { return WalkResult::interrupt(); })
      .wasInterrupted();
}

bool onlyReads(Operation& op) {
  return isMemoryEffectFree(&op) || hasSingleEffect<MemoryEffects::Read>(&op);
}

// Rolls the same-block elementwise consumers of `root`'s result, up to the
// tile stores that end them, into an scf.for over the rows of `scratch`, which
// the drain fills; a [1, 0]-transposed store is staged in `scratch` and copied
// out with strided loads. Changes nothing if a consumer does not qualify.
void tryEmitConsumerRowLoop(OpBuilder& b, Location loc, Operation* root,
                            Operation* firstAfterRoot,
                            const TileAddr& scratch) {
  Value rootRes = root->getResult(0);
  Block* block = root->getBlock();
  auto accTy = cast<VectorType>(rootRes.getType());
  int64_t m = accTy.getDimSize(0), n = accTy.getDimSize(1);

  SmallVector<Operation*> worklist = {root}, storeOps;
  DenseSet<Operation*> visited;
  SmallVector<std::pair<Value, TileAddr>> sinks;
  bool transposedOut = false;
  while (!worklist.empty()) {
    Operation* op = worklist.pop_back_val();
    if (op != root && !visited.insert(op).second) continue;
    if (op != root && (op->getBlock() != block || op->getNumResults() != 1 ||
                       !OpTrait::hasElementwiseMappableTraits(op)))
      return;
    Value res = op->getResult(0);
    bool isTr = false;
    auto [c, toErase] = matchTileStore(res, block, false, &isTr);
    if (c && (op != root || isTr)) {
      sinks.push_back({res, std::move(c)});
      llvm::append_range(storeOps, toErase);
      transposedOut |= isTr;
    }
    for (Operation* u : res.getUsers())
      if (!c || OpTrait::hasElementwiseMappableTraits(u)) worklist.push_back(u);
  }
  // A transposed sink is staged in the scratch tile of the root's element type.
  if (sinks.empty() ||
      (transposedOut &&
       (sinks.size() > 1 ||
        getElementTypeOrSelf(sinks[0].first) != accTy.getElementType())))
    return;
  llvm::sort(storeOps,
             [](Operation* a, Operation* b) { return a->isBeforeInBlock(b); });

  Operation* insertPt = storeOps.back();
  for (Operation* it = firstAfterRoot; it != insertPt; it = it->getNextNode())
    if (hasContraction(it) ||
        (!llvm::is_contained(storeOps, it) && !onlyReads(*it)))
      return;

  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(insertPt);
  auto cIdx = [&](int64_t v) {
    return arith::ConstantIndexOp::create(b, loc, v);
  };
  Value c0 = cIdx(0);
  int64_t colTile =
      transposedOut ? n : std::min<int64_t>(n, visited.size() > 36 ? 16 : 32);
  bool useColLoop = colTile < n && visited.size() > 36;
  SmallVector<scf::ForOp, 2> rowLoops;
  for (int64_t colOff = 0; colOff < (useColLoop ? colTile : n);
       colOff += colTile) {
    Value cOff = colOff == 0 ? c0 : cIdx(colOff);
    auto rowLoop = scf::ForOp::create(b, loc, c0, cIdx(m), cIdx(1));
    rowLoops.push_back(rowLoop);
    b.setInsertionPointToStart(rowLoop.getBody());
    Value r = rowLoop.getInductionVar();
    if (useColLoop) {
      auto colLoop = scf::ForOp::create(b, loc, c0, cIdx(n), cIdx(colTile));
      b.setInsertionPointToStart(colLoop.getBody());
      cOff = colLoop.getInductionVar();
    }
    auto rowIndices = [&](const TileAddr& a) {
      SmallVector<Value> idx = a.indices;
      idx.end()[-2] = arith::AddIOp::create(b, loc, idx.end()[-2], r);
      if (useColLoop || colOff != 0)
        idx.back() = arith::AddIOp::create(b, loc, idx.back(), cOff);
      return idx;
    };
    DenseMap<Value, Value> rowMap;
    rowMap[rootRes] = vector::LoadOp::create(
        b, loc, VectorType::get({colTile}, accTy.getElementType()),
        scratch.base, ValueRange{r, cOff});
    std::function<Value(Value)> sliceRow = [&](Value v) -> Value {
      if (auto it = rowMap.find(v); it != rowMap.end()) return it->second;
      auto vecTy = dyn_cast<VectorType>(v.getType());
      if (!vecTy || vecTy.getShape() != accTy.getShape()) return {};
      auto rowTy = VectorType::get({colTile}, vecTy.getElementType());
      Value res;
      Operation* def = v.getDefiningOp();
      auto tr = dyn_cast_or_null<vector::TransposeOp>(def);
      auto bcast = dyn_cast_or_null<vector::BroadcastOp>(
          tr && tr.getPermutation() == ArrayRef<int64_t>{1, 0}
              ? tr.getVector().getDefiningOp()
              : def);
      SplatElementsAttr splat;
      if (matchPattern(v, m_Constant(&splat))) {
        res = arith::ConstantOp::create(b, loc, splat.resizeSplat(rowTy));
      } else if (bcast) {
        auto srcVec = dyn_cast<VectorType>(bcast.getSourceType());
        Value src = bcast.getSource();
        if (!srcVec ||
            (srcVec.getRank() <= 1 && srcVec.getNumElements() == 1)) {
          res = vector::BroadcastOp::create(b, loc, rowTy, src);
        } else if (!tr && srcVec.getShape() == ArrayRef<int64_t>{n}) {
          if (TileAddr a = getLoadAddr(src, false, 1)) {
            SmallVector<Value> idx = a.indices;
            if (useColLoop || colOff != 0)
              idx.back() = arith::AddIOp::create(b, loc, idx.back(), cOff);
            res = vector::LoadOp::create(b, loc, rowTy, a.base, idx);
          } else if (colTile == n) {
            res = src;
          } else if (!useColLoop) {
            res = vector::ExtractStridedSliceOp::create(b, loc, src, colOff,
                                                        colTile, 1);
          }
        } else if (srcVec.getShape() ==
                   (tr ? ArrayRef<int64_t>{m} : ArrayRef<int64_t>{m, 1})) {
          res = vector::BroadcastOp::create(
              b, loc, rowTy, vector::ExtractOp::create(b, loc, src, r));
        }
      } else if (TileAddr a = getLoadAddr(v);
                 a && a.base != scratch.base && def->getBlock() == block &&
                 !def->isBeforeInBlock(firstAfterRoot) &&
                 llvm::all_of(sinks, [&](const auto& s) {
                   return a.base != s.second.base ||
                          a.indices == s.second.indices;
                 })) {
        res = vector::LoadOp::create(b, loc, rowTy, a.base, rowIndices(a));
      } else if (def && def->getBlock() == block && def->getNumResults() == 1 &&
                 OpTrait::hasElementwiseMappableTraits(def)) {
        IRMapping mapping;
        for (Value opnd : def->getOperands()) {
          Value s = sliceRow(opnd);
          if (!s) return {};
          mapping.map(opnd, s);
        }
        res = b.clone(*def, mapping)->getResult(0);
        res.setType(rowTy);
      }
      return res ? (rowMap[v] = res) : Value{};
    };
    for (const auto& [sinkVal, cOut] : sinks) {
      Value rowOut = sliceRow(sinkVal);
      if (!rowOut) {
        for (scf::ForOp rl : rowLoops) rl.erase();
        return;
      }
      vector::StoreOp::create(
          b, loc, rowOut, transposedOut ? scratch.base : cOut.base,
          transposedOut ? ValueRange{r, cOff} : ValueRange(rowIndices(cOut)));
    }
    for (Operation& op :
         llvm::make_early_inc_range(rowLoop.getBody()->without_terminator()))
      if (onlyReads(op) && llvm::all_of(op.getOperands(), [&](Value v) {
            return !rowLoop.getRegion().isAncestor(v.getParentRegion());
          }))
        op.moveBefore(rowLoop);
    b.setInsertionPointAfter(rowLoop);
  }
  if (transposedOut) {
    auto [sPtr, sStride] = extractPtrAndStride(b, loc, scratch);
    auto [cPtr, cStride] = extractPtrAndStride(b, loc, sinks[0].second);
    std::string asmStr = emitTileConfigAsm({}, 1, std::min<int64_t>(m, 16));
    for (int64_t mt = 0; mt < llvm::divideCeil(m, 16); ++mt)
      asmStr += llvm::formatv(
          "li t0, {0}\n\tadd t0, t0, $0\n\taddi t2, $2, {1}\n\tli t6, {2}\n\t"
          "{3}:\n\tvlse32.v v0, (t0), $1\n\tvse32.v v0, (t2)\n\t"
          "addi t0, t0, 4\n\tadd t2, t2, $3\n\taddi t6, t6, -1\n\t"
          "bnez t6, {3}b\n\t",
          mt * 64 * n, mt * 64, n, mt + 2);
    createInlineAsm(b, loc, asmStr, std::string("r,r,r,r,") + kClobbers,
                    {sPtr, sStride, cPtr, cStride});
    if (sinks[0].first == rootRes) rowLoops.back().erase();
  }
  for (Operation* op : llvm::reverse(storeOps)) op->erase();
}

struct CoralNPUMatrixCodegenPass
    : public impl::CoralNPUMatrixCodegenBase<CoralNPUMatrixCodegenPass> {
  void runOnOperation() override {
    auto funcOp = getOperation();
    if (!hasZvtTargetFeature(funcOp)) return;

    IRRewriter builder(&getContext());
    DenseMap<Type, TileAddr> scratchTiles;
    auto getScratchTile = [&](VectorType ty) {
      int64_t m = ty.getRank() == 1 ? 16 : llvm::alignTo(ty.getDimSize(0), 16);
      auto keyTy = VectorType::get({m, ty.getDimSize(ty.getRank() - 1)},
                                   ty.getElementType());
      TileAddr& tile = scratchTiles[keyTy];
      if (tile) return tile;
      Location loc = funcOp.getLoc();
      OpBuilder b = OpBuilder::atBlockBegin(&funcOp.getFunctionBody().front());
      Value c0 = arith::ConstantIndexOp::create(b, loc, 0);
      auto memTy = MemRefType::get(
          keyTy.getShape(), keyTy.getElementType(), MemRefLayoutAttrInterface{},
          iree_compiler::IREE::Codegen::WorkgroupLocalMemoryAttr::get(
              &getContext()));
      return tile = {memref::AllocOp::create(b, loc, memTy, ValueRange{},
                                             b.getI64IntegerAttr(64)),
                     {c0, c0}};
    };

    // Moves the op defining `v`, if in or after `target`, before it, operands
    // first; returns false if the op is `target` or has memory effects.
    std::function<bool(Value, Operation*)> hoist =
        [&](Value v, Operation* target) -> bool {
      if (auto arg = dyn_cast<BlockArgument>(v))
        return !target->isAncestor(arg.getOwner()->getParentOp());
      Operation* op = v.getDefiningOp();
      if (op == target) return false;
      if (!target->isProperAncestor(op) &&
          (op->getBlock() != target->getBlock() || op->isBeforeInBlock(target)))
        return true;
      if (!isMemoryEffectFree(op) ||
          !llvm::all_of(op->getOperands(),
                        [&](Value u) { return hoist(u, target); }))
        return false;
      op->moveBefore(target);
      return true;
    };
    auto hoistAddr = [&](const TileAddr& t, Operation* target) {
      return !t ||
             (hoist(t.base, target) && llvm::all_of(t.indices, [&](Value idx) {
                return hoist(idx, target);
              }));
    };

    // Reassemble per-row outer products emitted by strided 1D convolution
    // vectorization (M scalar-broadcast FMAs/muls inserted into a 2D
    // accumulator) into a single 2D vector.contract.
    funcOp.walk([&](scf::ForOp forOp) {
      if (forOp.getNumResults() != 1 || !forOp.getOps<scf::ForOp>().empty())
        return;
      Value iterArg = forOp.getRegionIterArgs()[0];
      auto accTy = dyn_cast<VectorType>(iterArg.getType());
      if (!accTy || accTy.getRank() != 2 || accTy.isScalable()) return;
      int64_t m = accTy.getDimSize(0), n = accTy.getDimSize(1);
      if (m < 3) return;
      auto yield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
      Value cur = yield.getOperand(0), rhs;
      SmallVector<Value> lhsScalars(m);
      auto getScalarSrc = [](Value v) -> Value {
        auto bc = stripExt(v).getDefiningOp<vector::BroadcastOp>();
        return bc && !isa<VectorType>(bc.getSourceType())
                   ? stripExt(bc.getSource())
                   : Value{};
      };
      for (int64_t r = m - 1; r >= 0; --r) {
        auto ins = cur.getDefiningOp<vector::InsertStridedSliceOp>();
        if (!ins || ins.hasNonUnitStrides() ||
            cast<IntegerAttr>(ins.getOffsets()[0]).getInt() != r ||
            cast<IntegerAttr>(ins.getOffsets()[1]).getInt() != 0)
          return;
        Value upd = ins.getValueToStore(), accRow, opA, opB;
        if (auto fma = upd.getDefiningOp<vector::FMAOp>()) {
          accRow = fma.getAcc();
          opA = fma.getLhs();
          opB = fma.getRhs();
        } else if (Operation* d = upd.getDefiningOp();
                   isa_and_nonnull<arith::AddFOp, arith::AddIOp>(d)) {
          int i = d->getOperand(0).getDefiningOp<vector::ExtractOp>() ? 1 : 0;
          Operation* mul = d->getOperand(i).getDefiningOp();
          if (!isa_and_nonnull<arith::MulFOp, arith::MulIOp>(mul)) return;
          accRow = d->getOperand(1 - i);
          opA = mul->getOperand(0);
          opB = mul->getOperand(1);
        } else {
          return;
        }
        auto ex = accRow.getDefiningOp<vector::ExtractOp>();
        if (!ex || ex.getSource() != iterArg ||
            ex.getStaticPosition() != ArrayRef<int64_t>{r})
          return;
        Value s = getScalarSrc(opB), rhsCur = opA;
        if (!s) {
          s = getScalarSrc(opA);
          rhsCur = opB;
        }
        if (!s || (r + 1 < m && (rhsCur != rhs ||
                                 s.getType() != lhsScalars[r + 1].getType())))
          return;
        rhs = rhsCur;
        lhsScalars[r] = s;
        cur = ins.getDest();
      }
      if (cur != iterArg) return;
      Location loc = yield.getLoc();
      builder.setInsertionPoint(yield);
      Type lhsElTy = lhsScalars[0].getType();
      Type rhsElTy = getElementTypeOrSelf(rhs.getType());
      Value lhs = vector::FromElementsOp::create(
          builder, loc, VectorType::get({m, 1}, lhsElTy), lhsScalars);
      if (!getLoadAddr(lhs)) {
        lhs.getDefiningOp()->erase();
        return;
      }
      if (lhsElTy != rhsElTy) {
        auto wideTy = VectorType::get({m, 1}, rhsElTy);
        lhs = isa<FloatType>(rhsElTy)
                  ? Value(arith::ExtFOp::create(builder, loc, wideTy, lhs))
                  : Value(arith::ExtSIOp::create(builder, loc, wideTy, lhs));
      }
      Value rhs2D = vector::ShapeCastOp::create(
          builder, loc, VectorType::get({1, n}, rhsElTy), rhs);
      AffineExpr d0, d1, d2;
      bindDims(&getContext(), d0, d1, d2);
      AffineExpr m0[] = {d0, d2}, m1[] = {d2, d1}, m2[] = {d0, d1};
      yield.setOperand(0, vector::ContractionOp::create(
                              builder, loc, lhs, rhs2D, iterArg,
                              ArrayRef<ArrayRef<AffineExpr>>{m0, m1, m2},
                              {vector::IteratorType::parallel,
                               vector::IteratorType::parallel,
                               vector::IteratorType::reduction}));
      (void)runRegionDCE(builder, forOp.getRegion());
    });

    SmallVector<vector::ContractionOp> contracts;
    funcOp.walk([&](vector::ContractionOp op) { contracts.push_back(op); });
    for (vector::ContractionOp contractOp : contracts) {
      bool swap = false;
      if (!isSupportedMatrixContraction(contractOp, swap) ||
          !contractOp->hasOneUse())
        continue;
      SmallVector<scf::ForOp> loopChain;
      Operation* root = contractOp;
      Value acc = contractOp.getAcc();
      // Each loop must carry only this accumulator, read only by the
      // contraction; enclosing loops may only read memory and hold no other
      // contraction.
      while (root->hasOneUse()) {
        auto yield = dyn_cast<scf::YieldOp>(*root->user_begin());
        auto p =
            yield ? dyn_cast<scf::ForOp>(yield->getParentOp()) : scf::ForOp{};
        if (!p || p.getNumResults() != 1 || acc != p.getRegionIterArgs()[0] ||
            !acc.hasOneUse() ||
            (!loopChain.empty() &&
             !llvm::all_of(
                 p.getBody()->without_terminator(), [&](Operation& o) {
                   return &o == root || (onlyReads(o) && !hasContraction(&o));
                 })))
          break;
        loopChain.push_back(p);
        root = p;
        acc = p.getInitArgs()[0];
      }
      scf::ForOp forOp = loopChain.empty() ? scf::ForOp{} : loopChain.front();
      TileAddr init = getLoadAddr(acc);
      if (root->use_empty() ||
          (root->hasOneUse() && isa<scf::YieldOp>(*root->user_begin())) ||
          !(init || matchPattern(acc, m_Zero()) ||
            matchPattern(acc, m_AnyZeroFloat())))
        continue;
      if (swap) {
        SmallVector<AffineMap> maps = contractOp.getIndexingMapsArray();
        Value lhs = contractOp.getLhs();
        contractOp.getLhsMutable().assign(contractOp.getRhs());
        contractOp.getRhsMutable().assign(lhs);
        std::swap(maps[0], maps[1]);
        contractOp.setIndexingMapsAttr(builder.getAffineMapArrayAttr(maps));
      }
      auto accTy = cast<VectorType>(contractOp.getType());
      int64_t rowsM = accTy.getRank() == 1 ? 1 : accTy.getDimSize(0);
      int64_t colsN = accTy.getDimSize(accTy.getRank() - 1);
      auto [c, opsToErase] = matchTileStore(
          root->getResult(0), root->getBlock(),
          hasCoralNPUTargetFeature(funcOp, "+zvfbfmin") && rowsM % 4 == 0);

      int64_t numTilesM = llvm::divideCeil(rowsM, 16);
      int64_t numTilesN = llvm::divideCeil(colsN, 16);
      Type inputElTy =
          getElementTypeOrSelf(traceToSourceValue(contractOp.getLhs()));
      Location loc = contractOp.getLoc();

      TileAddr lhs, rhs;
      int64_t kA = -1, kB = -1, kTripCount = 0;
      bool fuse = false;
      if (forOp) {
        // C is hoisted even when unfused: the drain goes right after the loop.
        if (!hoistAddr(c, root)) continue;
        lhs = getLoadAddr(contractOp.getLhs(), true);
        rhs = getLoadAddr(contractOp.getRhs(), true);
        Value iv = forOp.getInductionVar();
        splitKIndex(lhs, iv);
        splitKIndex(rhs, iv);
        kA = getKPos(lhs, iv);
        kB = getKPos(rhs, iv);
        // The asm K loop starts at the lower bound and advances k by one per
        // step.
        auto lb = getConstantIntValue(forOp.getLowerBound()),
             ub = getConstantIntValue(forOp.getUpperBound());
        if (lb && ub && isConstantIntValue(forOp.getStep(), 1))
          kTripCount = *ub - *lb;
        // Fusion erases the loop body, so its ops must be read-only.
        fuse = kA >= 0 && kB >= 0 && kTripCount > 0 &&
               llvm::all_of(forOp.getBody()->without_terminator(), onlyReads);
        if (fuse) {
          for (TileAddr* t : {&lhs, &rhs})
            llvm::replace(t->indices, iv, forOp.getLowerBound());
          fuse = hoistAddr(lhs, forOp) && hoistAddr(rhs, forOp);
        }
      }
      // The unfused asm takes full tiles of operand values as inputElTy.
      if (!fuse &&
          (rowsM % 16 != 0 || colsN % 16 != 0 ||
           getElementTypeOrSelf(stripExt(contractOp.getLhs())) != inputElTy ||
           getElementTypeOrSelf(stripExt(contractOp.getRhs())) != inputElTy))
        continue;

      TileAddr dst = c ? c : getScratchTile(accTy);
      Operation* firstAfterRoot = root->getNextNode();
      int64_t inBytes = inputElTy.getIntOrFloatBitWidth() / 8;
      int64_t tk = (kA != 1 || rowsM % 16 == 0) &&
                           (kB != 1 || colsN % 16 == 0) &&
                           kTripCount % (4 / inBytes) == 0
                       ? 4 / inBytes
                       : 1;
      bool hoistConfig =
          fuse && loopChain.size() > 1 &&
          llvm::all_of(llvm::drop_begin(loopChain), [&](scf::ForOp p) {
            return llvm::all_of(
                p.getBody()->without_terminator(), [&](Operation& o) {
                  return llvm::is_contained(loopChain, &o) ||
                         (o.getNumRegions() == 0 &&
                          llvm::none_of(o.getOperandTypes(),
                                        llvm::IsaPred<VectorType>) &&
                          llvm::none_of(o.getResultTypes(),
                                        llvm::IsaPred<VectorType>));
                });
          });
      if (fuse) {
        builder.setInsertionPoint(forOp);
        bool multi = loopChain.size() > 1;
        emitInlineFusedGEMM(builder, loc, lhs, rhs, multi ? TileAddr{} : dst,
                            multi ? TileAddr{} : init, numTilesM, numTilesN,
                            kTripCount, inputElTy, kA == 1, kB == 1, rowsM,
                            colsN, tk, hoistConfig);
      }
      if (!fuse || loopChain.size() > 1) {
        std::string inputConfig =
            hoistConfig && (!inputElTy.isF32() || (init && colsN < 16))
                ? emitTileConfigAsm(inputElTy, tk)
                : "";
        // Seeds (null `elTy`) or drains the tiles at `addr`.
        auto emitTileRows = [&](const TileAddr& addr, Type elTy) {
          createInlineAsm(builder, loc,
                          emitTileRowsAsm(numTilesM, numTilesN, "$0", "$1",
                                          elTy, rowsM, colsN) +
                              (elTy ? "" : inputConfig),
                          std::string("{a0},{a1},") + kClobbers,
                          extractPtrAndStride(builder, loc, addr));
        };
        // Seed the tiles, or zero them before a K loop, in a separate block;
        // otherwise the multiply block zeroes them.
        if (forOp || init) {
          builder.setInsertionPoint(root);
          if (init)
            emitTileRows(init, {});
          else
            createInlineAsm(builder, loc,
                            emitTileZeroAsm(numTilesM, numTilesN) + inputConfig,
                            "~{t5},~{t6}", {});
        }
        builder.setInsertionPoint(forOp || !c ? firstAfterRoot
                                              : opsToErase.back());
        emitTileRows(dst, getElementTypeOrSelf(dst.base));
      }
      if (!c) {
        if (accTy.getRank() == 2)
          tryEmitConsumerRowLoop(builder, loc, root, firstAfterRoot, dst);
        Value pad = arith::ConstantOp::create(
            builder, loc, builder.getZeroAttr(accTy.getElementType()));
        root->getResult(0).replaceAllUsesWith(vector::TransferReadOp::create(
            builder, loc, accTy, dst.base, dst.indices, pad,
            SmallVector<bool>(accTy.getRank(), true)));
      }
      for (Operation* op : llvm::reverse(opsToErase)) op->erase();
      if (fuse) {
        forOp.getResult(0).replaceAllUsesWith(forOp.getInitArgs()[0]);
        forOp.erase();
      } else {
        lowerContractionToAsm(builder, contractOp, numTilesM, numTilesN,
                              inputElTy, !forOp && !init);
      }
    }
    (void)runRegionDCE(builder, funcOp->getRegions());

    // Lower the contractions we did not take with OuterProduct: 0006 makes
    // LLVMCPUVirtualVectorLowering skip them and ConvertToLLVM would use Dot.
    if (hasContraction(funcOp)) {
      RewritePatternSet patterns(&getContext());
      vector::populateVectorToVectorCanonicalizationPatterns(patterns);
      vector::populateVectorContractLoweringPatterns(
          patterns, vector::VectorContractLowering::OuterProduct);
      if (failed(applyPatternsGreedily(funcOp, std::move(patterns))))
        return signalPassFailure();
    }
    // The vfwmaccbf16.vf asm below assumes VLEN = 128; keep in sync with
    // unrollAccumulatingInnerLoops (CoralNPUOptimizeVectorLoops).
    if (!hasCoralNPUTargetFeature(funcOp, "+zvfbfwma") ||
        getVlenFromTargetFeatures(funcOp).value_or(128) != 128)
      return;

    // Lower BF16 vecmat K loops (x[k] times row k of w into one or two
    // zero-initialized vector<32xf32> accumulators, an even trip count) to
    // vfwmaccbf16.vf, two K steps per iteration.

    auto v32f32 = VectorType::get({32}, builder.getF32Type());
    auto v32bf16 = VectorType::get({32}, builder.getBF16Type());
    funcOp.walk([&](scf::ForOp forOp) {
      int64_t n = forOp.getNumResults();
      auto lb = getConstantIntValue(forOp.getLowerBound()),
           ub = getConstantIntValue(forOp.getUpperBound());
      if ((n != 1 && n != 2) || !lb || !ub || (*ub - *lb) < 2 ||
          (*ub - *lb) % 2 != 0 || !isConstantIntValue(forOp.getStep(), 1) ||
          !forOp.getOps<scf::ForOp>().empty() ||
          !llvm::all_of(forOp.getBody()->without_terminator(), onlyReads) ||
          !llvm::all_of(forOp.getInitArgs(), [&](Value v) {
            return v.getType() == v32f32 && matchPattern(v, m_AnyZeroFloat());
          }))
        return;
      Value iv = forOp.getInductionVar();
      TileAddr xAddr, wAddr0;
      for (int64_t t = 0; t < n; ++t) {
        auto fma = forOp.getYieldedValues()[t].getDefiningOp<vector::FMAOp>();
        if (!fma || fma.getAcc() != forOp.getRegionIterArgs()[t]) return;
        auto getX = [&](Value v) -> TileAddr {
          v = traceToSourceValue(v);
          if (auto vt = dyn_cast<VectorType>(v.getType());
              vt && vt.getNumElements() != 1)
            return {};
          if (auto ex = v.getDefiningOp<vector::ExtractOp>();
              ex && llvm::all_of(ex.getStaticPosition(),
                                 [](int64_t i) { return i == 0; }))
            v = traceToSourceValue(ex.getSource());
          if (auto mld = v.getDefiningOp<memref::LoadOp>())
            return getRowMajorAddr(mld.getMemRef(), mld.getIndices(), 1);
          return getLoadAddr(v, true, 1);
        };
        Value l = fma.getLhs(), r = fma.getRhs();
        if (!getX(l) || !getLoadAddr(r, true)) std::swap(l, r);
        TileAddr x = getX(l), w = getLoadAddr(r, true);
        if (!x || !w || stripExt(r).getType() != v32bf16 ||
            !getElementTypeOrSelf(x.base).isBF16() || getKPos(x, iv) != 0 ||
            !getElementTypeOrSelf(w.base).isBF16() || getKPos(w, iv) != 1 ||
            (t > 0 &&
             (x.base != xAddr.base || x.indices != xAddr.indices ||
              w.base != wAddr0.base ||
              ArrayRef(w.indices).drop_back() !=
                  ArrayRef(wAddr0.indices).drop_back() ||
              getIndexDiff(w.indices.back(), wAddr0.indices.back()) != 32)))
          return;
        if (t == 0) std::tie(xAddr, wAddr0) = std::tie(x, w);
      }
      llvm::replace(xAddr.indices, iv, forOp.getLowerBound());
      llvm::replace(wAddr0.indices, iv, forOp.getLowerBound());
      if (!hoistAddr(xAddr, forOp) || !hoistAddr(wAddr0, forOp)) return;
      TileAddr scratch =
          getScratchTile(VectorType::get({n, 32}, builder.getF32Type()));
      Location loc = forOp.getLoc();
      builder.setInsertionPoint(forOp);
      Value xPtr = extractPtrAndStride(builder, loc, xAddr)[0];
      auto [wPtr, wStride] = extractPtrAndStride(builder, loc, wAddr0);
      Value dstPtr = extractPtrAndStride(builder, loc, scratch)[0];
      int64_t k = *ub - *lb, step = k % 8 == 0 ? 8 : (k % 4 == 0 ? 4 : 2);
      std::string asmStr = llvm::formatv(
          "li t4, 32\n\tvsetvli zero, t4, e32, m8, ta, ma\n\tvmv.v.i v16, 0\n\t"
          "{1}vsetvli zero, t4, e16, m4, ta, ma\n\tmv t2, $0\n\tmv t0, $1\n\t"
          "mv t1, $2\n\tli t6, {0}\n\t1:\n\t",
          k / step, n == 2 ? "vmv.v.i v24, 0\n\t" : "");
      for (int64_t s = 0; s < step; s += 2)
        asmStr += llvm::formatv(
            "flh fa0, {0}(t2)\n\t{2} v0, (t0)\n\tadd t0, t0, t1\n\t"
            "flh fa1, {1}(t2)\n\t{2} v8, (t0)\n\tadd t0, t0, t1\n\t"
            "vfwmaccbf16.vf v16, fa0, v0\n\t{3}"
            "vfwmaccbf16.vf v16, fa1, v8\n\t{4}",
            2 * s, 2 * s + 2, n == 2 ? "vl8re16.v" : "vle16.v",
            n == 2 ? "vfwmaccbf16.vf v24, fa0, v4\n\t" : "",
            n == 2 ? "vfwmaccbf16.vf v24, fa1, v12\n\t" : "");
      asmStr += llvm::formatv(
          "addi t2, t2, {0}\n\taddi t6, t6, -1\n\tbnez t6, 1b\n\t"
          "vsetvli zero, t4, e32, m8, ta, ma\n\tvse32.v v16, ($3)\n\t{1}",
          2 * step, n == 2 ? "addi t0, $3, 128\n\tvse32.v v24, (t0)\n\t" : "");
      createInlineAsm(
          builder, loc, asmStr,
          std::string("{a0},{a1},{a2},{a3},~{f10},~{f11},") + kClobbers,
          {xPtr, wPtr, wStride, dstPtr});
      Value c0 = scratch.indices[0];
      for (int64_t t = 0; t < n; ++t)
        forOp.getResult(t).replaceAllUsesWith(vector::LoadOp::create(
            builder, loc, v32f32, scratch.base,
            ValueRange{arith::ConstantIndexOp::create(builder, loc, t), c0}));
      forOp.erase();
    });
    (void)runRegionDCE(builder, funcOp->getRegions());
  }
};

}  // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createCoralNPUMatrixCodegenPass() {
  return std::make_unique<CoralNPUMatrixCodegenPass>();
}

}  // namespace mlir::coralnpu_compiler
