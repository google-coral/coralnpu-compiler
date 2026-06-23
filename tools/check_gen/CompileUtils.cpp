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

#include "tools/check_gen/CompileUtils.h"

// IREE headers
#include "iree/compiler/mlir_interop.h"

// MLIR headers
#include "mlir/CAPI/IR.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/Support/FileUtilities.h"

// LLVM headers
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::check_gen {

bool checkError(iree_compiler_error_t *error) {
  if (error) {
    const char *msg = ireeCompilerErrorGetMessage(error);
    llvm::errs() << "Compiler Error: " << msg << "\n";
    ireeCompilerErrorDestroy(error);
    return true;
  }
  return false;
}

LogicalResult writeModuleToFile(ModuleOp module, const std::string &outPath) {
  std::string error;
  auto outFile = openOutputFile(outPath, &error);
  if (!outFile) {
    llvm::errs() << error << "\n";
    return failure();
  }

  OpPrintingFlags printingFlags;
  printingFlags.assumeVerified();
  module.print(outFile->os(), printingFlags);

  outFile->keep();

  return success();
}

iree_compiler_session_t *initCompiler(MLIRContext *&context) {
  iree_compiler_session_t *session = ireeCompilerSessionCreate();

  std::vector<const char *> flags = {
      "--iree-hal-target-backends=vmvx",
  };
  if (checkError(
          ireeCompilerSessionSetFlags(session, flags.size(), flags.data()))) {
    ireeCompilerSessionDestroy(session);
    return nullptr;
  }

  MlirContext c_context = ireeCompilerSessionBorrowContext(session);
  if (!c_context.ptr) {
    llvm::errs() << "Failed to borrow context from session\n";
    ireeCompilerSessionDestroy(session);
    return nullptr;
  }
  context = unwrap(c_context);
  context->allowUnregisteredDialects();
  context->loadDialect<func::FuncDialect>();
  context->loadDialect<tensor::TensorDialect>();
  return session;
}

iree_compiler_output_t *compileModule(iree_compiler_invocation_t *inv,
                                      ModuleOp mergedModuleOp,
                                      void *&binaryData, uint64_t &binarySize) {
  MlirModule c_module = wrap(mergedModuleOp);
  MlirOperation c_op = mlirModuleGetOperation(c_module);

  if (!ireeCompilerInvocationImportBorrowModule(inv, c_op)) {
    llvm::errs() << "Failed to import module to invocation\n";
    return nullptr;
  }

  if (!ireeCompilerInvocationPipeline(inv, IREE_COMPILER_PIPELINE_STD)) {
    llvm::errs() << "Failed to run compilation pipeline\n";
    return nullptr;
  }

  iree_compiler_output_t *output = nullptr;
  if (checkError(ireeCompilerOutputOpenMembuffer(&output))) {
    return nullptr;
  }

  if (checkError(ireeCompilerInvocationOutputVMBytecode(inv, output))) {
    ireeCompilerOutputDestroy(output);
    return nullptr;
  }

  ireeCompilerOutputKeep(output);

  if (checkError(
          ireeCompilerOutputMapMemory(output, &binaryData, &binarySize))) {
    ireeCompilerOutputDestroy(output);
    return nullptr;
  }

  return output;
}

std::vector<TypedAttr> evaluateFunction(
    mlir::iree_compiler::ConstEval::CompiledBinary &binary, Location loc,
    StringRef funcName, ArrayRef<Type> resultTypes, ArrayRef<TypedAttr> args) {
  size_t numInputs = args.size();
  size_t numOutputs = resultTypes.size();

  mlir::iree_compiler::ConstEval::FunctionCall call(binary, numInputs,
                                                    numOutputs);
  if (failed(call.initialize(loc))) {
    llvm::errs() << "Failed to initialize call for " << funcName << "\n";
    return {};
  }

  for (auto attr : args) {
    if (failed(call.addArgument(loc, attr))) {
      llvm::errs() << "Failed to add argument to call for " << funcName << "\n";
      return {};
    }
  }

  if (failed(call.invoke(loc, funcName))) {
    llvm::errs() << "Failed to invoke " << funcName << "\n";
    return {};
  }

  std::vector<TypedAttr> outputs;
  outputs.reserve(numOutputs);

  for (size_t i = 0; i < numOutputs; ++i) {
    TypedAttr attr;
    if (failed(call.getResultAsAttr(loc, i, resultTypes[i], attr))) {
      llvm::errs() << "Failed to get result from call for " << funcName << "\n";
      return {};
    }
    outputs.push_back(attr);
  }

  return outputs;
}

}  // namespace mlir::check_gen
