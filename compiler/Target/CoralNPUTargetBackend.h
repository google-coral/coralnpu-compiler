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

#ifndef COMPILER_TARGET_CORALNPUTARGETBACKEND_H_
#define COMPILER_TARGET_CORALNPUTARGETBACKEND_H_

// IREE headers
#include "compiler/plugins/target/LLVMCPU/LLVMTargetOptions.h"
#include "compiler/plugins/target/LLVMCPU/LinkerTool.h"
#include "iree/compiler/Dialect/HAL/Target/TargetBackend.h"
#include "iree/compiler/Utils/OptionUtils.h"

// MLIR headers
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"

// LLVM headers
#include "llvm/Support/CommandLine.h"
#include "llvm/TargetParser/Triple.h"

// Standard C/C++ headers
#include <optional>
#include <string>
#include <vector>
namespace mlir {
class OpPassManager;
}  // namespace mlir

namespace mlir::coralnpu_compiler {

struct CoralNPUOptions {
  // CoralNPU specific options:
  int dtcmSizeKb = 8;
  int numVectorRegisters = 32;
  int tileParallelAlignment = 0;
  int tileVectorAlignment = 0;
  int tileUnrollAlignment = 0;
  int tileReductionAlignment = 1;

  // LLVMCPU options:
  std::string targetABI = "ilp32";
  std::string targetCPUFeatures = "+m,+f,+zvl128b,+zve32f";
  bool linkEmbedded = true;
  bool debugSymbols = false;
  std::string embeddedLinkerPath = "";

  void bindOptions(iree_compiler::OptionsBinder &binder) {
    static llvm::cl::OptionCategory category("CoralNPU HAL Target");

    binder.opt<std::string>(
        "coralnpu-target-abi", targetABI, llvm::cl::cat(category),
        llvm::cl::desc("LLVM target machine ABI; specify for -mabi"));

    binder.opt<std::string>("coralnpu-target-cpu-features", targetCPUFeatures,
                            llvm::cl::cat(category),
                            llvm::cl::desc("LLVM target machine CPU features"));

    binder.opt<bool>(
        "coralnpu-link-embedded", linkEmbedded, llvm::cl::cat(category),
        llvm::cl::desc("Links binaries into a platform-agnostic ELF"));

    binder.opt<std::string>(
        "coralnpu-embedded-linker-path", embeddedLinkerPath,
        llvm::cl::cat(category),
        llvm::cl::desc("Tool used to link embedded ELFs produced by CoralNPU"));

    binder.opt<bool>("coralnpu-debug-symbols", debugSymbols,
                     llvm::cl::cat(category),
                     llvm::cl::desc("Generate and embed debug information"));
    binder.opt<int>("coralnpu-dtcm-size-kb", dtcmSizeKb,
                    llvm::cl::cat(category),
                    llvm::cl::desc("Size of the DTCM in KB (default: 8)"));
    binder.opt<int>("coralnpu-num-vector-registers", numVectorRegisters,
                    llvm::cl::cat(category),
                    llvm::cl::desc("Number of vector registers (default: 32)"));
    binder.opt<int>(
        "coralnpu-tile-vector-alignment", tileVectorAlignment,
        llvm::cl::cat(category),
        llvm::cl::desc(
            "Tile alignment for vector parallel loops (0 for auto)"));
    binder.opt<int>(
        "coralnpu-tile-unroll-alignment", tileUnrollAlignment,
        llvm::cl::cat(category),
        llvm::cl::desc(
            "Tile alignment for unrolled parallel loops (0 for auto)"));
    binder.opt<int>(
        "coralnpu-tile-reduction-alignment", tileReductionAlignment,
        llvm::cl::cat(category),
        llvm::cl::desc("Tile alignment for reduction loops (default: 1)"));
    binder.opt<int>(
        "coralnpu-tile-parallel-alignment", tileParallelAlignment,
        llvm::cl::cat(category),
        llvm::cl::desc(
            "Tile alignment for generic parallel loops (0 for auto)"));
  }

  LogicalResult validate(MLIRContext *context = nullptr) const;
};

class CoralNPUTargetBackend final
    : public iree_compiler::IREE::HAL::TargetBackend {
 public:
  explicit CoralNPUTargetBackend(const CoralNPUOptions &options);

  std::string getLegacyDefaultDeviceID() const override;

  void getDefaultExecutableTargets(
      MLIRContext *context, StringRef deviceID, DictionaryAttr deviceConfigAttr,
      SmallVectorImpl<iree_compiler::IREE::HAL::ExecutableTargetAttr>
          &executableTargetAttrs) const override;

  iree_compiler::IREE::HAL::ExecutableTargetAttr getExecutableTarget(
      MLIRContext *context,
      const iree_compiler::IREE::HAL::LLVMTarget &target) const;

  void getDependentDialects(DialectRegistry &registry) const override;

  void buildConfigurationPassPipeline(
      iree_compiler::IREE::HAL::ExecutableTargetAttr targetAttr,
      OpPassManager &passManager) override;

  void buildTranslationPassPipeline(
      iree_compiler::IREE::HAL::ExecutableTargetAttr targetAttr,
      OpPassManager &passManager) override;

  void buildLinkingPassPipeline(OpPassManager &passManager) override;

  LogicalResult serializeExecutable(
      const SerializationOptions &options,
      iree_compiler::IREE::HAL::ExecutableVariantOp variantOp,
      OpBuilder &executableBuilder) override;

 private:
  std::optional<iree_compiler::IREE::HAL::LLVMTarget> getVariantTarget(
      iree_compiler::IREE::HAL::ExecutableVariantOp variantOp);

  LogicalResult serializeStaticLibraryExecutable(
      const SerializationOptions &options,
      const iree_compiler::IREE::HAL::LLVMTarget &target,
      iree_compiler::IREE::HAL::ExecutableVariantOp variantOp,
      OpBuilder &executableBuilder, const std::string &libraryName,
      const std::string &queryFunctionName,
      const SmallVector<iree_compiler::IREE::HAL::Artifact> &objectFiles);

  LogicalResult serializeDynamicLibraryExecutable(
      const SerializationOptions &options,
      const iree_compiler::IREE::HAL::LLVMTarget &target,
      iree_compiler::IREE::HAL::ExecutableVariantOp variantOp,
      OpBuilder &executableBuilder, const std::string &libraryName,
      const llvm::Triple &targetTriple,
      const SmallVector<iree_compiler::IREE::HAL::Artifact> &objectFiles,
      iree_compiler::IREE::HAL::LinkerTool *linkerTool);

  iree_compiler::IREE::HAL::LLVMTargetOptions defaultOptions_;
  CoralNPUOptions options_;
};

}  // namespace mlir::coralnpu_compiler

#endif  // COMPILER_TARGET_CORALNPUTARGETBACKEND_H_
