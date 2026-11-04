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

#include "metrics/compiler/MetricsAttr.h"

// MLIR:
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OperationSupport.h"

namespace mlir::coralnpu_compiler {

Attribute getMetricsEntry(Operation *funcOp, llvm::StringRef key) {
  auto dictAttr = funcOp->getAttrOfType<DictionaryAttr>(kMetricsAttrName);
  return dictAttr ? dictAttr.get(key) : Attribute();
}

void setMetricsEntry(Operation *funcOp, llvm::StringRef key, Attribute value) {
  NamedAttrList entries;
  if (auto dictAttr = funcOp->getAttrOfType<DictionaryAttr>(kMetricsAttrName)) {
    entries.append(dictAttr.getValue());
  }
  entries.set(key, value);
  funcOp->setAttr(kMetricsAttrName,
                  entries.getDictionary(funcOp->getContext()));
}

}  // namespace mlir::coralnpu_compiler
