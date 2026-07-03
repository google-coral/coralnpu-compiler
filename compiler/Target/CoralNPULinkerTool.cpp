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

#include "compiler/Target/CoralNPULinkerTool.h"

#include <cstdlib>
#include <optional>
#include <string>

#include "iree/compiler/Utils/StringUtils.h"
#include "iree/compiler/Utils/ToolUtils.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "coralnpu-linker"

namespace mlir::coralnpu_compiler {
namespace {

using LinkerTool = iree_compiler::IREE::HAL::LinkerTool;
using Artifact = iree_compiler::IREE::HAL::Artifact;
using Artifacts = iree_compiler::IREE::HAL::Artifacts;
using LLVMTargetOptions = iree_compiler::IREE::HAL::LLVMTargetOptions;

class CoralNPULinkerTool final : public LinkerTool {
 public:
  CoralNPULinkerTool(const llvm::Triple& targetTriple,
                     LLVMTargetOptions targetOptions)
      : LinkerTool(targetTriple, std::move(targetOptions)) {}

  std::string getLinkerPath() const {
    if (!targetOptions.embeddedLinkerPath.empty()) {
      return targetOptions.embeddedLinkerPath;
    }

#ifdef CORALNPU_LINKER_PATH
    const std::string configuredPath = CORALNPU_LINKER_PATH;
    if (!configuredPath.empty()) {
      return configuredPath;
    }
#endif

    std::string toolPath =
        mlir::iree_compiler::findTool("riscv32-unknown-elf-ld");
    if (!toolPath.empty()) {
      return toolPath;
    }

    llvm::errs() << "error: required `riscv32-unknown-elf-ld` was not found "
                    "after searching:\n"
                 << " * configured LLVM target linker path\n"
                 << " * CMake-configured CoralNPU linker path\n"
                 << " * system PATH\n";

    return "";
  }

  std::string getLinkerScriptPath() const {
    std::string path = "";
#ifdef CORALNPU_LINKER_SCRIPT_PATH
    path = CORALNPU_LINKER_SCRIPT_PATH;
#endif
    if (path.empty()) {
      return "";
    }

    if (llvm::sys::fs::exists(path)) {
      return path;
    }

    if (!llvm::sys::path::is_relative(path)) {
      return path;
    }

    // Try to resolve relative to executable
    std::string mainExecutablePath =
        llvm::sys::fs::getMainExecutable(nullptr, nullptr);
    if (mainExecutablePath.empty()) {
      return path;
    }

    llvm::SmallString<256> resolvedPath(mainExecutablePath);
    llvm::sys::path::remove_filename(resolvedPath);  // to compiler/tools/
    llvm::sys::path::append(resolvedPath, "..", "..", path);  // to crt/
    if (!llvm::sys::fs::exists(resolvedPath)) {
      return path;
    }
    llvm::sys::path::remove_dots(resolvedPath, /*remove_dot_dot=*/true);
    return std::string(resolvedPath);
  }

  mlir::LogicalResult configureModule(
      llvm::Module* llvmModule,
      llvm::ArrayRef<llvm::Function*> exportedFuncs) override {
    for (auto& llvmFunc : *llvmModule) {
      llvmFunc.addFnAttr("nonlazybind");
      llvmFunc.setUWTableKind(llvm::UWTableKind::None);
    }

    if (targetOptions.target.debugSymbols) {
      for (auto* llvmFunc : exportedFuncs) {
        llvmFunc->setVisibility(
            llvm::GlobalValue::VisibilityTypes::DefaultVisibility);
        llvmFunc->setLinkage(llvm::GlobalValue::LinkageTypes::ExternalLinkage);
        llvmFunc->setUWTableKind(llvm::UWTableKind::Default);
      }
    }

    return mlir::success();
  }

  std::optional<Artifacts> linkDynamicLibrary(
      llvm::StringRef libraryName,
      llvm::ArrayRef<Artifact> objectFiles) override {
    Artifacts artifacts;

    if (!objectFiles.empty()) {
      artifacts.libraryFile =
          Artifact::createVariant(objectFiles.front().path, "so");
    } else {
      artifacts.libraryFile = Artifact::createTemporary(libraryName, "so");
    }
    artifacts.libraryFile.close();

    const std::string linkerPath = getLinkerPath();
    if (linkerPath.empty()) {
      return std::nullopt;
    }

    const std::string linkerScriptPath = getLinkerScriptPath();
    if (linkerScriptPath.empty()) {
      llvm::errs() << "error: CoralNPU linker script path was not configured\n";
      return std::nullopt;
    }

    llvm::SmallString<256> archiveDirectory(linkerScriptPath);
    llvm::sys::path::remove_filename(archiveDirectory);

    llvm::SmallString<256> ireeArchivePath(archiveDirectory);
    llvm::sys::path::append(ireeArchivePath, "libcoralnpu_iree.a");

    llvm::SmallString<256> crtArchivePath(archiveDirectory);
    llvm::sys::path::append(crtArchivePath, "libcoralnpu_crt.a");

    const std::string ireeArchivePathString(ireeArchivePath);
    const std::string crtArchivePathString(crtArchivePath);

    llvm::SmallVector<std::string> flags = {
        mlir::iree_compiler::escapeCommandLineComponent(linkerPath),
        "-o " + mlir::iree_compiler::escapeCommandLineComponent(
                    artifacts.libraryFile.path),
        "--build-id=none",
        "-nostdlib",
        "-static",
        "-T " +
            mlir::iree_compiler::escapeCommandLineComponent(linkerScriptPath),
        "--no-undefined",
        "--gc-sections",
        "--discard-all",
        "--no-warn-rwx-segments",
    };

    LLVM_DEBUG(llvm::SmallString<256> mapFilePath(artifacts.libraryFile.path);
               llvm::sys::path::replace_extension(mapFilePath, "map");
               flags.push_back("-Map=" + std::string(mapFilePath)););

    if (!targetOptions.target.debugSymbols) {
      flags.push_back("--strip-debug");
    }

    for (const auto& objectFile : objectFiles) {
      flags.push_back(
          mlir::iree_compiler::escapeCommandLineComponent(objectFile.path));
    }

    flags.push_back("--whole-archive");

    flags.push_back(
        mlir::iree_compiler::escapeCommandLineComponent(ireeArchivePathString));

    flags.push_back(
        mlir::iree_compiler::escapeCommandLineComponent(crtArchivePathString));

    flags.push_back("--no-whole-archive");

    if (mlir::failed(runLinkCommand(llvm::join(flags, " "), ""))) {
      if (targetOptions.keepLinkerArtifacts) {
        for (auto& objectFile : objectFiles) {
          if (objectFile.outputFile) {
            llvm::errs() << "CoralNPU linker input preserved: "
                         << objectFile.outputFile->getFilename() << "\n";
            objectFile.keep();
          }
        }
      }

      return std::nullopt;
    }

    if (targetOptions.keepLinkerArtifacts) {
      artifacts.keepAllFiles();

      for (auto& objectFile : objectFiles) {
        objectFile.keep();
      }
    }

    return artifacts;
  }

 private:
  CoralNPUOptions coralNPUOptions_;
};

}  // namespace

std::unique_ptr<iree_compiler::IREE::HAL::LinkerTool> createCoralNPULinkerTool(
    const llvm::Triple& targetTriple,
    iree_compiler::IREE::HAL::LLVMTargetOptions& targetOptions) {
  return std::make_unique<CoralNPULinkerTool>(targetTriple, targetOptions);
}

}  // namespace mlir::coralnpu_compiler
