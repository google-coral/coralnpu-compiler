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

// MLIR:
#include "mlir/IR/Attributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LogicalResult.h"

// LLVM:
#include "llvm/ADT/ArrayRef.h"

namespace mlir::coralnpu_compiler {

// Parses VLEN from target CPU features (+zvl<N>b).
FailureOr<int64_t> getVlenFromTargetFeatures(FunctionOpInterface funcOp);

// Helper to create a tiling level attribute with default (false) scalable
// flags.
Attribute getTilingLevelAttr(MLIRContext *context, ArrayRef<int64_t> sizes);

}  // namespace mlir::coralnpu_compiler

#endif  // COMPILER_TRANSFORMS_CORALNPUTILESIZESELECTIONUTILS_H_
