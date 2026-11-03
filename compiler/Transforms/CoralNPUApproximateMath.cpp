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

#include "compiler/Transforms/Passes.h"

// MLIR:
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

// LLVM:
#include "llvm/Support/MathExtras.h"

// Standard C/C++ headers
#include <cmath>

namespace mlir::coralnpu_compiler {

#define GEN_PASS_DEF_CORALNPUAPPROXIMATEMATH
#include "compiler/Transforms/Passes.h.inc"

namespace {

/// Builds the f32 (or vector of f32) arithmetic of an approximation to an
/// elementwise op on f32, f16 or bf16: the operand is extended to f32 and the
/// result truncated back to the type of the op.
struct F32ApproximationBuilder {
  F32ApproximationBuilder(RewriterBase &rewriter, Location loc, Type resultType)
      : rewriter(rewriter), loc(loc), resultType(resultType) {
    f32Type = rewriter.getF32Type();
    i32Type = rewriter.getI32Type();
    if (auto vecType = dyn_cast<VectorType>(resultType)) {
      f32Type = vecType.clone(f32Type);
      i32Type = vecType.clone(i32Type);
    }
  }

  /// Returns true for f32, f16 and bf16 scalars and vectors.
  static bool supports(Type type) {
    if (isa<ShapedType>(type) && !isa<VectorType>(type)) return false;
    Type elementType = getElementTypeOrSelf(type);
    return elementType.isF32() || elementType.isBF16() || elementType.isF16();
  }

  /// Returns a constant of `type` with value `attr`, splat for vectors.
  Value splat(Type type, TypedAttr attr) {
    if (auto vecType = dyn_cast<VectorType>(type))
      attr = SplatElementsAttr::get(vecType, attr);
    return arith::ConstantOp::create(rewriter, loc, type, attr);
  }
  /// Returns the f32 constant `value`, shaped like the result.
  Value cst(float value) {
    return splat(f32Type, rewriter.getF32FloatAttr(value));
  }
  /// Returns `a * b`.
  Value mul(Value a, Value b) {
    return arith::MulFOp::create(rewriter, loc, a, b);
  }
  /// Returns `a + b`.
  Value add(Value a, Value b) {
    return arith::AddFOp::create(rewriter, loc, a, b);
  }
  /// Evaluates the polynomial with coefficients `coeffs`, highest degree
  /// first, at `x` (Horner's method).
  Value horner(Value x, ArrayRef<float> coeffs) {
    Value acc = cst(coeffs.front());
    for (float c : coeffs.drop_front()) acc = add(mul(acc, x), cst(c));
    return acc;
  }
  /// Returns `x` clamped to [lo, hi]; a NaN `x` gives `hi`.
  Value clamp(Value x, float lo, float hi) {
    return arith::MaxNumFOp::create(
        rewriter, loc, arith::MinNumFOp::create(rewriter, loc, x, cst(hi)),
        cst(lo));
  }
  /// Returns `x` extended to f32 (unchanged for an f32 result).
  Value widen(Value x) {
    if (resultType == f32Type) return x;
    return arith::ExtFOp::create(rewriter, loc, f32Type, x,
                                 arith::FastMathFlagsAttr{});
  }
  /// Returns the f32 `x` truncated to the result type (unchanged for an f32
  /// result).
  Value narrow(Value x) {
    if (resultType == f32Type) return x;
    return arith::TruncFOp::create(rewriter, loc, resultType, x);
  }

  RewriterBase &rewriter;
  Location loc;
  Type resultType;
  Type f32Type;
  Type i32Type;
};

/// Approximates tanh(x) with a [7/6] minimax rational function of the input
/// clamped to [-6.35, 6.35] (max absolute error ~6.1e-6).
Value createFastTanh(F32ApproximationBuilder &b, Value x) {
  Value xc = b.clamp(x, -6.3473315f, 6.3473315f);
  Value x2 = b.mul(xc, xc);
  Value num = b.horner(x2, {1.0f, 519.88947f, 26767.564f, 214877.48f});
  Value den = b.horner(x2, {34.0196f, 4667.8545f, 98392.99f, 214877.48f});
  return arith::DivFOp::create(b.rewriter, b.loc, b.mul(xc, num), den);
}

/// Approximates exp(x) by range reduction to r = x - n ln(2), |r| <= ln(2)/2,
/// and a degree-6 polynomial in r (max relative error ~3e-7). The result
/// overflows to +inf where exp does (x > 88.7228) and is 0 for x < -86.99
/// (where exp(x) < 1.7e-38).
Value createFastExp(F32ApproximationBuilder &b, Value x) {
  // Clamp so that the rounded exponent n stays in [-126, 128]: the lower
  // bound is -126 ln(2); the upper one is past ln(FLT_MAX) = 88.7228 so that
  // larger inputs still overflow below.
  Value xc = b.clamp(x, -87.33655f, 88.8f);
  // Adding 1.5 * 2^23 + 126 = 12583038.0f (0x4b40007e) rounds xc * log2(e)
  // to the nearest integer n and places n + 126, the biased exponent of
  // 2^(n-1), in bits 7..0; subtracting it back recovers n exactly.
  Value magic = b.cst(12583038.0f);
  Value z = b.add(b.mul(xc, b.cst(1.44269504f)), magic);
  Value n = arith::SubFOp::create(b.rewriter, b.loc, z, magic);
  Value r = b.add(b.add(xc, b.mul(n, b.cst(-0.69314575f))),
                  b.mul(n, b.cst(-1.4286068e-6f)));
  Value p = b.horner(
      r, {0.0013888889f, 0.0083333338f, 0.041666668f, 0.16666667f, 0.5f});
  Value q = b.add(r, b.mul(b.mul(p, r), r));
  // Scale by 2^(n-1) and then by 2: 2^n is already inf for n = 128, where
  // exp(x) can still be finite.
  Value pow2nm1 = arith::BitcastOp::create(
      b.rewriter, b.loc, b.f32Type,
      arith::ShLIOp::create(
          b.rewriter, b.loc,
          arith::BitcastOp::create(b.rewriter, b.loc, b.i32Type, z),
          b.splat(b.i32Type, b.rewriter.getI32IntegerAttr(23))));
  return b.mul(b.mul(b.add(q, b.cst(1.0f)), pow2nm1), b.cst(2.0f));
}

/// Approximates erf(x) with Abramowitz & Stegun 7.1.28 (max absolute error
/// ~1.4e-6 in f32):
///   erf(x) = copysign(1 - (1 + a1 |x| + ... + a6 |x|^6)^-16, x).
Value createFastErf(F32ApproximationBuilder &b, Value x) {
  Value ax = arith::MinNumFOp::create(
      b.rewriter, b.loc, math::AbsFOp::create(b.rewriter, b.loc, x),
      b.cst(4.0f));
  Value p = b.horner(ax, {0.0000430638f, 0.0002765672f, 0.0001520143f,
                          0.0092705272f, 0.0422820123f, 0.0705230784f, 1.0f});
  for (int i = 0; i < 4; ++i) p = b.mul(p, p);
  Value one = b.cst(1.0f);
  Value mag = arith::SubFOp::create(
      b.rewriter, b.loc, one, arith::DivFOp::create(b.rewriter, b.loc, one, p));
  return math::CopySignOp::create(b.rewriter, b.loc, mag, x);
}

/// Replaces `OpTy` on f32, f16 and bf16 with `approximate` computed in f32.
template <typename OpTy, Value (*approximate)(F32ApproximationBuilder &, Value)>
struct FastApproxPattern : OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    if (!F32ApproximationBuilder::supports(op.getType())) return failure();
    F32ApproximationBuilder b(rewriter, op.getLoc(), op.getType());
    rewriter.replaceOp(op, b.narrow(approximate(b, b.widen(op.getOperand()))));
    return success();
  }
};

/// Matches the f32 expansion of chlo.erfc(x) by StableHLO's
/// ChloLegalizeToStablehlo (jax.nn.gelu(approximate=False) uses erfc),
///   select(|x| < 1, 1 - x * T(x^2), select(x < 0, 2 - E, E)),
/// where the constant term of the polynomial T is 2/sqrt(pi) and E
/// approximates erfc(|x|) for |x| >= 1, and replaces it with 1 - erf(x)
/// computed as in createFastErf.
struct FastApproxErfcExpansionPattern : OpRewritePattern<arith::SelectOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(arith::SelectOp op,
                                PatternRewriter &rewriter) const override {
    if (!F32ApproximationBuilder::supports(op.getType())) return failure();
    // select(|x| < 1, 1 - x * T(x^2), ...).
    auto absLtOne = op.getCondition().getDefiningOp<arith::CmpFOp>();
    auto oneSubErf = op.getTrueValue().getDefiningOp<arith::SubFOp>();
    if (!absLtOne || absLtOne.getPredicate() != arith::CmpFPredicate::OLT ||
        !matchPattern(absLtOne.getRhs(), m_OneFloat()) || !oneSubErf ||
        !matchPattern(oneSubErf.getLhs(), m_OneFloat())) {
      return failure();
    }
    auto absX = absLtOne.getLhs().getDefiningOp<math::AbsFOp>();
    auto xMulT = oneSubErf.getRhs().getDefiningOp<arith::MulFOp>();
    if (!absX || !xMulT) return failure();
    Value x = absX.getOperand();
    Value t = xMulT.getLhs() == x   ? xMulT.getRhs()
              : xMulT.getRhs() == x ? xMulT.getLhs()
                                    : Value();
    auto tAdd = t ? t.getDefiningOp<arith::AddFOp>() : nullptr;
    APFloat t0(0.0f);
    if (!tAdd || !matchPattern(tAdd.getRhs(), m_ConstantFloat(&t0)) ||
        std::abs(t0.convertToDouble() - 2.0 * llvm::numbers::inv_sqrtpi) >
            1e-5) {
      return failure();
    }
    // ... select(x < 0, 2 - E, E).
    auto negSelect = op.getFalseValue().getDefiningOp<arith::SelectOp>();
    if (!negSelect) return failure();
    auto xLtZero = negSelect.getCondition().getDefiningOp<arith::CmpFOp>();
    auto twoSubE = negSelect.getTrueValue().getDefiningOp<arith::SubFOp>();
    APFloat two(0.0f);
    if (!xLtZero || xLtZero.getPredicate() != arith::CmpFPredicate::OLT ||
        xLtZero.getLhs() != x ||
        !matchPattern(xLtZero.getRhs(), m_AnyZeroFloat()) || !twoSubE ||
        twoSubE.getRhs() != negSelect.getFalseValue() ||
        !matchPattern(twoSubE.getLhs(), m_ConstantFloat(&two)) ||
        !two.isExactlyValue(2.0)) {
      return failure();
    }
    F32ApproximationBuilder b(rewriter, op.getLoc(), op.getType());
    Value res = arith::SubFOp::create(rewriter, b.loc, b.cst(1.0f),
                                      createFastErf(b, b.widen(x)));
    rewriter.replaceOp(op, b.narrow(res));
    return success();
  }
};

struct CoralNPUApproximateMathPass final
    : impl::CoralNPUApproximateMathBase<CoralNPUApproximateMathPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FastApproxPattern<math::TanhOp, createFastTanh>,
                 FastApproxPattern<math::ExpOp, createFastExp>,
                 FastApproxPattern<math::ErfOp, createFastErf>,
                 FastApproxErfcExpansionPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<Pass> createCoralNPUApproximateMathPass() {
  return std::make_unique<CoralNPUApproximateMathPass>();
}

}  // namespace mlir::coralnpu_compiler
