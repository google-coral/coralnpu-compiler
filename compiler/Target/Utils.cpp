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

#include "compiler/Target/Utils.h"

#include "mlir/IR/Builders.h"

namespace mlir::coralnpu_compiler {

iree_compiler::IREE::HAL::TargetBackend::SupportedTypes
getCoralNPUSupportedTypes(MLIRContext *context) {
  iree_compiler::IREE::HAL::TargetBackend::SupportedTypes s;
  Builder b(context);

  // Scalar types:

  // s.addScalarType(b.getF8E8M0Type());
  s.addScalarType(b.getBF16Type());
  // s.addScalarType(b.getF16Type());
  // s.addScalarType(b.getTF32Type());
  s.addScalarType(b.getF32Type());
  // s.addScalarType(b.getF64Type());
  // s.addScalarType(b.getF80Type());
  // s.addScalarType(b.getF128Type());

  s.addScalarType(b.getIndexType());

  // s.addScalarType(b.getI1Type());
  // s.addScalarType(b.getI2Type());
  // s.addScalarType(b.getI4Type());
  s.addScalarType(b.getI8Type());
  s.addScalarType(b.getI16Type());
  s.addScalarType(b.getI32Type());
  // s.addScalarType(b.getI64Type());

  // Element types:

  // s.addElementType(b.getF8E8M0Type());
  // s.addElementType(b.getBF16Type());
  // s.addElementType(b.getF16Type());
  // s.addElementType(b.getTF32Type());
  s.addElementType(b.getF32Type());
  // s.addElementType(b.getF64Type());
  // s.addElementType(b.getF80Type());
  // s.addElementType(b.getF128Type());

  s.addElementType(b.getIndexType());

  // s.addElementType(b.getI1Type());
  // s.addElementType(b.getI2Type());
  // s.addElementType(b.getI4Type());
  s.addElementType(b.getI8Type());
  s.addElementType(b.getI16Type());
  s.addElementType(b.getI32Type());
  // s.addElementType(b.getI64Type());

  return s;
}

}  // namespace mlir::coralnpu_compiler
