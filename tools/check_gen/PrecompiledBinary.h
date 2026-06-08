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

#ifndef CORALNPU_COMPILER_TOOLS_CHECK_GEN_PRECOMPILED_BINARY_H_
#define CORALNPU_COMPILER_TOOLS_CHECK_GEN_PRECOMPILED_BINARY_H_

// IREE headers
#include "iree/compiler/ConstEval/Runtime.h"
#include "iree/vm/api.h"

// MLIR headers
#include "mlir/Support/LLVM.h"

// Standard C/C++ headers
#include <string>

namespace mlir::check_gen {

// Class to manage precompiled VM bytecode binary for ConstEval JIT execution.
class PrecompiledBinary
    : public mlir::iree_compiler::ConstEval::CompiledBinary {
public:
  LogicalResult init(Location loc, const void *data, size_t length);
  ~PrecompiledBinary() override;

  iree_vm_module_t *getModule();

private:
  std::string buffer;
};

} // namespace mlir::check_gen

#endif // CORALNPU_COMPILER_TOOLS_CHECK_GEN_PRECOMPILED_BINARY_H_
