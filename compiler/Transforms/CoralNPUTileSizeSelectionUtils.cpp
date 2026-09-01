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

// IREE:
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"

// MLIR:
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"

// LLVM:
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;
using namespace mlir::iree_compiler;

namespace mlir::coralnpu_compiler {

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

Attribute getTilingLevelAttr(MLIRContext *context, ArrayRef<int64_t> sizes) {
  SmallVector<bool> scalableFlags(sizes.size(), false);
  return IREE::Codegen::LoweringConfigTilingLevelAttr::get(
      context, sizes, /*tileInterchange=*/{}, scalableFlags);
}

bool hasZvtTargetFeature(Operation *op) {
  if (!op) return false;
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(op);
  return targetAttr && targetAttr.getBackend().getValue() == "coralnpu" &&
         hasFeature(targetAttr.getConfiguration(), "+zvtbase");
}

// INVARIANT: three predicates gate the Zvt path at successive pipeline stages,
// each accepting a superset of the next. Relaxing one requires re-checking the
// others.
//   1. here (linalg): selects 16x16 tile sizes.
//   2. isZvtMatrixContraction() in iree-v3.11.0-0010-*.patch (vector, loose):
//      suppresses IREE's generic contract lowering. Upstream, so it tests
//      "+zvtbase" but not the backend name.
//   3. isSupportedMatrixContraction() in CoralNPUMatrixCodegen.cpp (vector,
//      strictest): the one that actually emits Zvt asm.
bool isZvtMatrixContraction(Operation *op) {
  if (!hasZvtTargetFeature(op)) return false;

  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp || !linalg::isaContractionOpInterface(linalgOp) ||
      linalgOp.getNumDpsInputs() != 2 || linalgOp.getNumDpsInits() != 1) {
    return false;
  }

  auto in0Shaped =
      dyn_cast<ShapedType>(linalgOp.getDpsInputOperand(0)->get().getType());
  auto in1Shaped =
      dyn_cast<ShapedType>(linalgOp.getDpsInputOperand(1)->get().getType());
  auto initShaped =
      dyn_cast<ShapedType>(linalgOp.getDpsInitOperand(0)->get().getType());
  if (!in0Shaped || !in1Shaped || !initShaped || !initShaped.hasStaticShape()) {
    return false;
  }

  Type in0El = in0Shaped.getElementType();
  Type in1El = in1Shaped.getElementType();
  Type accEl = initShaped.getElementType();
  if (in0El != in1El) return false;
  if (!(accEl.isF32() && in0El.isF32()) &&
      !(accEl.isInteger(32) && in0El.isInteger(8))) {
    return false;
  }

  if (isa<linalg::Mmt4DOp>(linalgOp)) {
    if (!in0Shaped.hasStaticShape() || !in1Shaped.hasStaticShape() ||
        initShaped.getRank() != 4) {
      return false;
    }
    return initShaped.getDimSize(2) == 16 && initShaped.getDimSize(3) == 16 &&
           in0Shaped.getDimSize(3) == 1;
  }

  if (linalgOp.getNumParallelLoops() != 2 ||
      linalgOp.getNumReductionLoops() != 1 || initShaped.getRank() != 2) {
    return false;
  }
  int64_t m = initShaped.getDimSize(0), n = initShaped.getDimSize(1);
  return m >= 16 && m % 16 == 0 && n >= 16 && n % 16 == 0;
}

}  // namespace mlir::coralnpu_compiler
