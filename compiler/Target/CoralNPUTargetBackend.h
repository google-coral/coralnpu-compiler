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
#include "iree/compiler/Codegen/Utils/CodegenOptions.h"
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

class RegisterAllocationReportCollector;

struct CoralNPUOptions {
  // CoralNPU specific options:
  int dtcmSizeKb = 32;
  std::string linkerScriptPath = "";
  int numVectorRegisters = 32;
  int maxLoopUnrolling = 32;
  std::string registerAllocationReportFormat = "none";
  std::string registerAllocationReportDir = "";
  std::string registerAllocationReportFilter = ".*dispatch.*|main";
  bool linkExecutables = false;

  // LLVMCPU options:
  std::string targetABI = "ilp32";
  std::string targetCPUFeatures = "+m,+f,+zvl128b,+zve32f";
  bool linkEmbedded = true;
  bool debugSymbols = false;
  bool keepLinkerArtifacts = false;
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

    binder.opt<bool>(
        "coralnpu-link-executables", linkExecutables, llvm::cl::cat(category),
        llvm::cl::desc("Link all executables into a single library (default: "
                       "false, to fit within ITCM)"));

    binder.opt<std::string>(
        "coralnpu-embedded-linker-path", embeddedLinkerPath,
        llvm::cl::cat(category),
        llvm::cl::desc("Tool used to link embedded ELFs produced by CoralNPU"));

    binder.opt<std::string>(
        "coralnpu-linker-script-path", linkerScriptPath,
        llvm::cl::cat(category),
        llvm::cl::desc("Path to the linker script used to link embedded ELFs"));

    binder.opt<bool>("coralnpu-debug-symbols", debugSymbols,
                     llvm::cl::cat(category),
                     llvm::cl::desc("Generate and embed debug information"));
    binder.opt<bool>(
        "coralnpu-keep-linker-artifacts", keepLinkerArtifacts,
        llvm::cl::cat(category),
        llvm::cl::desc("Keep LLVM linker target artifacts (.so/.elf/etc)"));
    binder.opt<int>("coralnpu-dtcm-size-kb", dtcmSizeKb,
                    llvm::cl::cat(category),
                    llvm::cl::desc("Size of the DTCM in KB (default: 32)"));
    binder.opt<int>("coralnpu-num-vector-registers", numVectorRegisters,
                    llvm::cl::cat(category),
                    llvm::cl::desc("Number of vector registers (default: 32)"));

    binder.opt<int>(
        "coralnpu-max-loop-unrolling", maxLoopUnrolling,
        llvm::cl::cat(category),
        llvm::cl::desc(
            "Maximum unroll factor allowed for any loop (default: 32). LLVM "
            "unrolling can be too aggressive, which leads to ITCM overflow. "
            "Setting this to 1 effectively disables unrolling."));

    binder.opt<std::string>(
        "coralnpu-dump-register-allocation-report-format",
        registerAllocationReportFormat, llvm::cl::cat(category),
        llvm::cl::desc(
            "Register allocation report format (none, pretty, json)"));
    binder.opt<std::string>(
        "coralnpu-dump-register-allocation-report-dir",
        registerAllocationReportDir, llvm::cl::cat(category),
        llvm::cl::desc("Directory to dump register allocation reports. "
                       "'-' for stdout, empty for stderr."));
    binder.opt<std::string>(
        "coralnpu-dump-register-allocation-report-filter",
        registerAllocationReportFilter, llvm::cl::cat(category),
        llvm::cl::desc("Regex pattern to filter functions in the report "
                       "(default: '.*dispatch.*|main')"));
  }

  LogicalResult validate(MLIRContext *context = nullptr) const;
};

class CoralNPUTargetBackend final
    : public iree_compiler::IREE::HAL::TargetBackend {
 public:
  explicit CoralNPUTargetBackend(const CoralNPUOptions &options);

  std::string getLegacyDefaultDeviceID() const override;

  SupportedTypes getSupportedTypes(MLIRContext *context) const override;

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
  iree_compiler::CPUCodegenOptions codegenOptions_;
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

  void dumpRegisterAllocationReport(
      const RegisterAllocationReportCollector &collector,
      iree_compiler::IREE::HAL::ExecutableVariantOp variantOp);

  iree_compiler::IREE::HAL::LLVMTargetOptions defaultOptions_;
  CoralNPUOptions options_;
};

}  // namespace mlir::coralnpu_compiler

#endif  // COMPILER_TARGET_CORALNPUTARGETBACKEND_H_
