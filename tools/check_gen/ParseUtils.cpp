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

#include "tools/check_gen/ParseUtils.h"

// MLIR headers
#include "mlir/AsmParser/AsmParser.h"

// LLVM headers
#include "llvm/Support/raw_ostream.h"

namespace mlir::check_gen {

std::vector<std::vector<int64_t>> parseInstance(llvm::StringRef raw) {
  std::vector<std::vector<int64_t>> res;
  size_t i = 0;
  while (i < raw.size()) {
    if (raw[i] != '(') {
      llvm::errs() << "Expected '(' at position " << i << " in " << raw << "\n";
      return {};
    }
    size_t start = i + 1;
    size_t end = raw.find(')', start);
    if (end == llvm::StringRef::npos) {
      llvm::errs() << "Expected ')' after position " << start << " in " << raw
                   << "\n";
      return {};
    }
    llvm::StringRef groupRaw = raw.slice(start, end);

    std::vector<int64_t> group;
    if (!groupRaw.empty()) {
      llvm::SmallVector<llvm::StringRef, 4> tokens;
      groupRaw.split(tokens, ',');
      for (auto token : tokens) {
        int64_t val;
        token = token.trim();
        if (token.getAsInteger(10, val)) {
          llvm::errs() << "Invalid integer: " << token << " in group "
                       << groupRaw << "\n";
          return {};
        }
        group.push_back(val);
      }
    }
    res.push_back(group);

    i = end + 1;
  }
  return res;
}

std::vector<std::vector<std::vector<int64_t>>>
parseAllInstances(const llvm::cl::list<std::string> &rawInstances) {
  std::vector<std::vector<std::vector<int64_t>>> instances;
  for (const auto &raw : rawInstances) {
    auto inst = parseInstance(raw);
    if (inst.empty()) {
      return {};
    }
    instances.push_back(inst);
  }
  return instances;
}

Type parseTypeFromDecl(llvm::StringRef decl, MLIRContext *context) {
  // Find "-> "
  size_t arrowPos = decl.find("->");
  if (arrowPos == llvm::StringRef::npos)
    return nullptr;

  // Find "(" after "->"
  size_t openParen = decl.find('(', arrowPos);
  if (openParen == llvm::StringRef::npos)
    return nullptr;

  // Find ":" after "("
  size_t colon = decl.find(':', openParen);
  if (colon == llvm::StringRef::npos)
    return nullptr;

  // Find ")" after ":"
  size_t closeParen = decl.find(')', colon);
  if (closeParen == llvm::StringRef::npos)
    return nullptr;

  llvm::StringRef typeStr = decl.slice(colon + 1, closeParen);

  // Strip attributes if present (e.g. {some_attr})
  size_t brace = typeStr.find('{');
  if (brace != llvm::StringRef::npos) {
    typeStr = typeStr.slice(0, brace);
  }

  typeStr = typeStr.trim();

  return parseType(typeStr, context);
}

} // namespace mlir::check_gen
