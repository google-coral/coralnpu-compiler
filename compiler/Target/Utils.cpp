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

// IREE:
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"

// MLIR:
#include "mlir/IR/Builders.h"

// LLVM:
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

// Standard library:
#include <algorithm>
#include <cassert>
#include <system_error>

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
  s.addElementType(b.getBF16Type());
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

std::mutex &getReportMutex() {
  static std::mutex mutex;
  return mutex;
}

void writeReport(Operation *op, StringRef dir, StringRef fileName,
                 llvm::function_ref<void(llvm::raw_ostream &)> writeFn) {
  if (dir.empty() || dir == "-") {
    std::lock_guard<std::mutex> lock(getReportMutex());
    llvm::raw_ostream &os = (dir == "-") ? llvm::outs() : llvm::errs();
    writeFn(os);
    os.flush();
    return;
  }

  if (std::error_code ec = llvm::sys::fs::create_directories(dir)) {
    op->emitWarning() << "failed to create report directory '" << dir
                      << "': " << ec.message();
    return;
  }
  llvm::SmallString<128> filePath(dir);
  llvm::sys::path::append(filePath, fileName);
  std::error_code ec;
  llvm::raw_fd_ostream os(filePath, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    op->emitWarning() << "failed to open report file '" << filePath
                      << "': " << ec.message();
    return;
  }
  writeFn(os);
  os.close();
  if (os.has_error()) {
    op->emitWarning() << "failed to write report file '" << filePath
                      << "': " << os.error().message();
    os.clear_error();
  }
}

FailureOr<int64_t> getVlenFromTargetFeatures(FunctionOpInterface funcOp) {
  auto targetAttr =
      iree_compiler::IREE::HAL::ExecutableTargetAttr::lookup(funcOp);
  if (!targetAttr) return failure();
  return getVlenFromTargetFeatures(targetAttr.getConfiguration());
}

FailureOr<int64_t> getVlenFromTargetFeatures(DictionaryAttr config) {
  if (!config) return failure();
  auto attr = config.getAs<StringAttr>("cpu_features");
  if (!attr) return failure();

  // +zvl<N>b implies every smaller +zvl<M>b, so VLEN is the largest N.
  int64_t vlen = 0;
  SmallVector<StringRef> features;
  attr.getValue().split(features, ',');
  for (StringRef feature : features) {
    feature = feature.trim();
    int64_t n = 0;
    if (feature.consume_front("+zvl") && feature.consume_back("b") &&
        !feature.getAsInteger(/*Radix=*/10, n)) {
      vlen = std::max(vlen, n);
    }
  }
  if (vlen == 0) return failure();
  return vlen;
}

int64_t getNativeVectorBytes(Operation *op) {
  // One 128-bit register, if the target does not give the size.
  constexpr int64_t kDefaultNativeVectorBytes = 16;
  auto target = iree_compiler::IREE::HAL::ExecutableTargetAttr::lookup(op);
  if (!target || !target.getConfiguration()) return kDefaultNativeVectorBytes;
  auto size =
      target.getConfiguration().getAs<IntegerAttr>("native_vector_size");
  return size ? size.getInt() : kDefaultNativeVectorBytes;
}

bool hasCoralNPUTargetFeature(Operation *op, StringRef feature) {
  auto targetAttr = iree_compiler::IREE::HAL::ExecutableTargetAttr::lookup(op);
  return targetAttr && targetAttr.getBackend().getValue() == "coralnpu" &&
         iree_compiler::hasFeature(targetAttr.getConfiguration(), feature);
}

bool hasZvtTargetFeature(Operation *op) {
  return hasCoralNPUTargetFeature(op, "+zvtbase");
}

constexpr StringLiteral kZvtMatmulMnemonics[] = {"vtmms.tvv", "vtfmm.alt.tvv",
                                                 "vtfmm.tvv"};

StringRef getZvtMatmulMnemonic(Type elTy) {
  if (elTy.isInteger(8) || elTy.isInteger(16)) return kZvtMatmulMnemonics[0];
  if (elTy.isBF16()) return kZvtMatmulMnemonics[1];
  assert(elTy.isF32() && "expected i8, i16, bf16, or f32 Zvt input type");
  return kZvtMatmulMnemonics[2];
}

bool containsZvtMatmulMnemonic(StringRef asmStr) {
  return llvm::any_of(kZvtMatmulMnemonics, [&](StringRef mnemonic) {
    return asmStr.contains(mnemonic);
  });
}

}  // namespace mlir::coralnpu_compiler
