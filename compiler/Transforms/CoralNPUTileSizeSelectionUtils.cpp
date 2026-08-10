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
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"

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

}  // namespace mlir::coralnpu_compiler
