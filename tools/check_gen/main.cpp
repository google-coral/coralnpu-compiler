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

// CoralNPU headers
#include "tools/check_gen/CheckTestGenerator.h"
#include "tools/check_gen/CompileUtils.h"
#include "tools/check_gen/ParseUtils.h"

// IREE headers
#include "iree/compiler/embedding_api.h"

// MLIR headers
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"

// LLVM headers
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

// Standard C/C++ headers
#include <string>
#include <vector>

using namespace mlir;
using namespace mlir::check_gen;

// --- Command Line Arguments ---

static llvm::cl::list<std::string> inputFilenames(
    llvm::cl::Positional,
    llvm::cl::desc("<test_fn.mlir> <gen_fn1.vmfb/mlir> ..."),
    llvm::cl::OneOrMore);

static llvm::cl::opt<std::string> outputDir("o",
                                            llvm::cl::desc("Output directory"),
                                            llvm::cl::value_desc("dir"),
                                            llvm::cl::init("."));

static llvm::cl::list<std::string> instancesRaw(
    "instance", llvm::cl::desc("Instance shapes, e.g. (4,8)(8,4)"),
    llvm::cl::OneOrMore);

static llvm::cl::opt<std::string> defaultGen(
    "default-gen", llvm::cl::desc("Default generators file (vmfb or mlir)"),
    llvm::cl::init(""));

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "IREE check template generator\n");

  ireeCompilerGlobalInitialize();

  // Make sure the output directory exists
  if (auto ec = llvm::sys::fs::create_directories(outputDir)) {
    llvm::errs() << "Failed to create output directory " << outputDir << ": "
                 << ec.message() << "\n";
    ireeCompilerGlobalShutdown();
    return 1;
  }

  // Parse instances
  auto instances = parseAllInstances(instancesRaw);
  if (instances.empty()) {
    ireeCompilerGlobalShutdown();
    return 2;
  }

  if (inputFilenames.empty()) {
    llvm::errs() << "Error: Must specify at least one test function file\n";
    ireeCompilerGlobalShutdown();
    return 2;
  }

  // Initialize Compiler Session and Context
  MLIRContext *context = nullptr;
  iree_compiler_session_t *session = initCompiler(context);
  if (!session) {
    ireeCompilerGlobalShutdown();
    return 1;
  }

  // Run the generator
  bool success = false;
  {
    // Enforce scope to ensure `generator` (and its MLIR/IREE members) is
    // destroyed BEFORE the compiler session/context is destroyed. Otherwise,
    // destroying MLIR/IREE objects after the context is gone causes
    // use-after-free crashes.
    CheckTestGenerator generator(context, session, std::move(instances),
                                 inputFilenames, outputDir, defaultGen);
    success = generator.run();
  }

  // Cleanup session (destroys context)
  ireeCompilerSessionDestroy(session);
  ireeCompilerGlobalShutdown();

  return success ? 0 : 1;
}
