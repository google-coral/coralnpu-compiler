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

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"

namespace cl = llvm::cl;

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  static llvm::cl::OptionCategory listerCategory("MLIR Op Lister Options");

  cl::opt<std::string> dialectFilter(
      cl::Positional, cl::desc("<dialect namespace (e.g. linalg)>"),
      cl::Optional, cl::cat(listerCategory));

  cl::HideUnrelatedOptions(listerCategory);
  cl::ParseCommandLineOptions(
      argc, argv,
      "MLIR Dialect Operation Lister\n\n"
      "  This tool lists all registered operations of a given dialect in the "
      "MLIR context.\n"
      "  If no dialect namespace is provided, it lists all registered "
      "operations across all dialects.\n");

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);

  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  if (dialectFilter.getNumOccurrences() > 0) {
    std::string dialectName = dialectFilter.getValue();
    auto ops = context.getRegisteredOperationsByDialect(dialectName);
    if (ops.empty()) {
      llvm::errs() << "No operations found for dialect: " << dialectName
                   << "\n";
      return 1;
    }
    for (auto op : ops) {
      llvm::outs() << op.getStringRef() << "\n";
    }
  } else {
    auto ops = context.getRegisteredOperations();
    for (auto op : ops) {
      llvm::outs() << op.getStringRef() << "\n";
    }
  }

  return 0;
}
