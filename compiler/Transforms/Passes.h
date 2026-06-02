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

#ifndef COMPILER_TRANSFORMS_PASSES_H_
#define COMPILER_TRANSFORMS_PASSES_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::coralnpu_compiler {

// Registers all CoralNPU passes.
void registerCoralNPUPasses();

#define GEN_PASS_DECL
#include "compiler/Transforms/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createCoralNPUAffinityAnnotationPass();
std::unique_ptr<OperationPass<ModuleOp>>
createCoralNPUAffinityAnnotationPass(CoralNPUAffinityAnnotationOptions options);

} // namespace mlir::coralnpu_compiler

#endif // COMPILER_TRANSFORMS_PASSES_H_
