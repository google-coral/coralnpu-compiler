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

// The `coralnpu.metrics` attribute, through which the metrics passes attach
// per-dispatch facts to the dispatch functions for the metrics bundle.

#ifndef METRICS_COMPILER_METRICSATTR_H_
#define METRICS_COMPILER_METRICSATTR_H_

// MLIR:
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"

// LLVM:
#include "llvm/ADT/StringRef.h"

namespace mlir::coralnpu_compiler {

// Name of the discardable attribute through which the metrics passes attach
// per-dispatch metrics to the dispatch function. The value is a
// DictionaryAttr; the metrics bundle copies its entries into the entry of the
// dispatch (MetricsBundle.h). Supported values: integers, floats, strings,
// booleans, unit, and arrays and dictionaries of those.
inline constexpr llvm::StringLiteral kMetricsAttrName = "coralnpu.metrics";

// Returns the entry |key| of the kMetricsAttrName attribute of |funcOp|, or
// null if there is none.
Attribute getMetricsEntry(Operation *funcOp, llvm::StringRef key);

// Sets the entry |key| of the kMetricsAttrName attribute of |funcOp| to
// |value|, keeping the other entries.
void setMetricsEntry(Operation *funcOp, llvm::StringRef key, Attribute value);

}  // namespace mlir::coralnpu_compiler

#endif  // METRICS_COMPILER_METRICSATTR_H_
