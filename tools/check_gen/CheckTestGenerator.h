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

#ifndef CORALNPU_COMPILER_TOOLS_CHECK_GEN_CHECK_TEST_GENERATOR_H_
#define CORALNPU_COMPILER_TOOLS_CHECK_GEN_CHECK_TEST_GENERATOR_H_

// CoralNPU headers
#include "tools/check_gen/PrecompiledBinary.h"

// IREE headers
#include "iree/compiler/embedding_api.h"

// MLIR headers
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"

// LLVM headers
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"

// Standard C/C++ headers
#include <memory>
#include <string>
#include <vector>

namespace mlir::check_gen {

// Orchestrates the generation of check tests from a test template and
// generators.
class CheckTestGenerator {
public:
  CheckTestGenerator(
      MLIRContext *context, iree_compiler_session_t *session,
      std::vector<std::vector<std::vector<int64_t>>> instances,
      std::vector<std::string> inputFiles, llvm::StringRef outputDir,
      llvm::StringRef defaultGenPath);

  // Runs the generation process. Returns true on success.
  bool run();

private:
  // Information about a generator (either MLIR or VMFB).
  struct GeneratorInfo {
    std::string filename;
    std::string funcName;
    size_t numArguments = 0;
    std::shared_ptr<PrecompiledBinary> binary;
    std::string callingConvention;
  };

  // Loads test function and generators, merges them into mergedModuleOp.
  bool parseAndMerge();

  // Parses a single MLIR file and merges its function into the symbol table.
  func::FuncOp parseAndMergeFunc(StringRef filename, SymbolTable &symbolTable);

  // Validates that generator inputs/outputs match the test function and
  // instances.
  bool validateInputs();

  // Processes a single instance.
  bool processInstance(size_t instIdx);

  // Evaluates generators for a given instance to produce input attributes.
  std::vector<TypedAttr>
  evaluateGenerators(const std::vector<std::vector<int64_t>> &instance);

  // Refines the shapes of the test function based on input attributes.
  OwningOpRef<ModuleOp> refineShapes(const std::vector<TypedAttr> &inputAttrs,
                                     size_t instIdx);

  // Evaluates the refined test function to produce expected output attributes.
  std::vector<TypedAttr>
  evaluateRefinedTest(ModuleOp refinedTestModuleOp, func::FuncOp checkTestFunc,
                      const std::vector<TypedAttr> &inputAttrs, size_t instIdx);

  // Generates the final check test MLIR module.
  bool generateCheckTest(func::FuncOp refinedTestFunc,
                         const std::vector<TypedAttr> &inputAttrs,
                         const std::vector<TypedAttr> &outputAttrs,
                         const std::vector<std::vector<int64_t>> &instance);

  // Helper to add constant inputs to the check function.
  LogicalResult addConstantInputs(OpBuilder &funcBuilder, Location loc,
                                  const std::vector<TypedAttr> &inputAttrs,
                                  ArrayRef<Type> refinedArgTypes,
                                  std::vector<Value> &callArgs);

  // Helper to add assertions to the check function.
  LogicalResult addAssertions(OpBuilder &funcBuilder, Location loc,
                              func::CallOp callOp,
                              const std::vector<TypedAttr> &outputAttrs);

  // jetski: change all the std::string to llvm::StringRef (or std::string_view)
  // Helper to inline and cleanup the check module.
  LogicalResult inlineAndCleanup(ModuleOp outModule,
                                 llvm::StringRef testFuncName);

  bool loadGenerator(llvm::StringRef path, Type expectedType,
                     GeneratorInfo &gen);

  // Loads a default generator for a given type.
  bool loadDefaultGenerator(Type expectedType, GeneratorInfo &gen);

  // Helper to get or load generator binary from path.
  std::shared_ptr<PrecompiledBinary> getOrLoadBinary(llvm::StringRef path,
                                                     Location loc);



  MLIRContext *context;
  iree_compiler_session_t *session;
  std::vector<std::vector<std::vector<int64_t>>> instances;
  std::vector<std::string> inputFiles;
  std::string outputDir;

  OwningOpRef<ModuleOp> mergedModuleOp;
  func::FuncOp testFunc;

  std::vector<GeneratorInfo> generators;

  std::string defaultGenPath;
  llvm::StringMap<std::shared_ptr<PrecompiledBinary>> binaryCache;
};

} // namespace mlir::check_gen

#endif // CORALNPU_COMPILER_TOOLS_CHECK_GEN_CHECK_TEST_GENERATOR_H_
