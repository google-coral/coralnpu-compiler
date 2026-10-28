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

#ifndef COMPILER_TRANSFORMS_CORALNPUTILESIZESELECTIONUTILS_H_
#define COMPILER_TRANSFORMS_CORALNPUTILESIZESELECTIONUTILS_H_

// IREE:
#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUTypes.h"

// MLIR:
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Support/LogicalResult.h"

// LLVM:
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::coralnpu_compiler {

// The root op of a dispatch and its loops, for the DTCM tile-size selection.
struct DTCMRoot {
  Operation *op = nullptr;  // has the strategy config
  iree_compiler::IREE::CPU::LoweringConfigAttr config;
  SmallVector<int64_t> loopRanges;          // static ranges
  SmallVector<unsigned> parallelLoops;      // outermost first
  SmallVector<unsigned> reductionLoops;     // outermost first
  SmallVector<TilingInterface> nonRootOps;  // multi-op reduction peers
};

// Classifies loops of `tilingOp` into parallel and reduction loops (outermost
// first).
void classifyLoops(TilingInterface tilingOp,
                   SmallVectorImpl<unsigned> &parallelLoops,
                   SmallVectorImpl<unsigned> &reductionLoops);

// Returns true if `op`'s first DPS input has a narrower integer/float bit width
// (>= 8 bits) than `op`'s first result.
bool hasWideningInput(linalg::LinalgOp op);

// Infers the convolution dimensions of `op` if it is a named or generic
// convolution or pooling (including 3/4-input quantized ops).
FailureOr<linalg::ConvolutionDimensions> inferConvDims(Operation *op);

// Parses VLEN from target CPU features: the largest N among +zvl<N>b.
FailureOr<int64_t> getVlenFromTargetFeatures(FunctionOpInterface funcOp);
FailureOr<int64_t> getVlenFromTargetFeatures(DictionaryAttr config);

// Helper to create a tiling level attribute with default (false) scalable
// flags.
Attribute getTilingLevelAttr(MLIRContext *context, ArrayRef<int64_t> sizes);

// Returns the DTCM space in bytes that the buffers of one DTCM tile can use:
// the DTCM size minus the stack and the reserve for static data, rounded down
// to a multiple of 4 KB. Returns 0 if less than 4 KB is left, if the DTCM size
// is not positive, or if the stack or the reserve is negative.
int64_t getDTCMTileBudgetBytes(int64_t dtcmSizeKb, int64_t stackSizeKb,
                               int64_t reserveKb);

// The lowering config entry of a root op that asks CoralNPUPromoteOperandsPass
// to copy the tiles of the operands to DTCM. Its value is the list of the
// operand numbers of the root op (as the promote_operands of the IREE GPU
// configs). The DTCM pass lists all the operands, and the promotion pass only
// checks that the list is not empty: it copies the tiles of all operands of
// the root op and of the ops fused with it.
inline constexpr llvm::StringLiteral kPromoteOperandsName = "promote_operands";

// Returns true if `loweringConfig` is a CPU lowering config with a
// kPromoteOperandsName entry that is a list with at least one operand.
bool hasPromoteOperands(Attribute loweringConfig);

// Returns the native vector size in bytes of the HAL executable target of `op`
// (the "native_vector_size" entry of the target config), or 16 bytes if the
// target does not give it.
int64_t getNativeVectorBytes(Operation *op);

// Returns true if the enclosing HAL executable targets CoralNPU with +zvtbase.
bool hasZvtTargetFeature(Operation *op);

// Returns true if `op` targets CoralNPU with `feature` (e.g. "+zvfbfmin").
bool hasCoralNPUTargetFeature(Operation *op, StringRef feature);

// Returns true for the Zvt element types: f32 x f32 or bf16 x bf16 -> f32, and
// i8 x i8 -> i32.
bool isZvtElementTypes(Type lhs, Type rhs, Type acc);

// Returns true if the shape of the accumulator of a vector.contract (after the
// unit dims are dropped) is a shape that CoralNPUMatrixCodegen lowers to Zvt
// asm: `m` rows and `n` columns, or, if `rank1`, one dim of `n` elements along
// N (then `m` is 1). `swap` is true if the right-hand side indexes the rows.
// The other conditions of the asm (element types, a K of 1, the K loop, the
// init) are not part of this check. isSupportedMatrixContraction in
// CoralNPUMatrixCodegen.cpp uses this check.
bool isZvtAccumulatorShape(int64_t m, int64_t n, bool rank1, bool swap);

// Returns true if the Zvt asm can load the two inputs of `op`, a linalg op
// with two inputs and one init, from memory with Zvt element types. The asm
// loads an input that no linalg op computes. It also loads the input of an
// elementwise op that computes an input from one tensor with only an
// arith.extf or arith.extsi (the asm reads the narrow elements), if no linalg
// op computes that tensor. Other producers (e.g. a dequantization or an
// arith.sitofp) keep the asm from taking `op`. The element types that the asm
// loads (before the extension) and the init element type must be Zvt element
// types (isZvtElementTypes); e.g. a bf16 lhs that an arith.extf widens to f32
// does not go with an f32 rhs. Keep in sync with traceToSourceValue and
// isSupportedMatrixContraction in CoralNPUMatrixCodegen.cpp.
bool isZvtReadableInputs(Operation *op);

// Returns true if hasZvtTargetFeature(op) and op is a Zvt-eligible contraction
// or convolution: supported types, shapes and layouts, a zero or loaded init,
// and no reduction consumer.
bool isZvtMatrixContraction(Operation *op);

// isZvtMatrixContraction without the target feature check.
bool isZvtMatrixContractionShape(Operation *op);

// Returns true if `op` is a contraction with extent > 1 (or dynamic) along
// both M and N, excluding zero-reuse matvec, vecmat, and dot products.
bool hasContractionReuse(Operation *op);

}  // namespace mlir::coralnpu_compiler

#endif  // COMPILER_TRANSFORMS_CORALNPUTILESIZESELECTIONUTILS_H_
