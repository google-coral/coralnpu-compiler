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

#include "compiler/Target/CoralNPUTargetBackend.h"

#include "compiler/Target/CoralNPULinkerTool.h"
#include "compiler/Transforms/Passes.h"

// IREE headers
#include "compiler/plugins/target/LLVMCPU/Builtins/Device.h"
#include "compiler/plugins/target/LLVMCPU/Builtins/Musl.h"
#include "compiler/plugins/target/LLVMCPU/Builtins/UKernel.h"
#include "compiler/plugins/target/LLVMCPU/LLVMIRPasses.h"
#include "compiler/plugins/target/LLVMCPU/LibraryBuilder.h"
#include "compiler/plugins/target/LLVMCPU/StaticLibraryGenerator.h"
#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUDialect.h"
#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUTypes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"
#include "iree/compiler/Dialect/HAL/Utils/LLVMLinkerUtils.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/compiler/Utils/ModuleUtils.h"

// MLIR headers
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

// LLVM headers
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"

#define DEBUG_TYPE "coralnpu-target"
using llvm::dbgs;

using namespace mlir;
using namespace mlir::iree_compiler;

namespace mlir::coralnpu_compiler {
namespace {

static void dumpLLVMModuleToPath(StringRef path, StringRef baseName,
                                 StringRef suffix, StringRef extPrefix,
                                 llvm::Module &module) {
  // Dump disassembly to path.
  llvm::SmallVector<char> textData;
  llvm::raw_svector_ostream textOstream(textData);

  module.print(textOstream, nullptr);
  std::string textExtension = extPrefix.str() + ".ll";
  IREE::HAL::dumpDataToPath(path, baseName, suffix, textExtension,
                            StringRef(textData.data(), textData.size()));

  // Dump bitcode to path.
  llvm::SmallVector<char> binaryData;
  llvm::raw_svector_ostream binaryOstream(binaryData);
  // Write the specified module to the specified output stream.
  llvm::WriteBitcodeToFile(module, binaryOstream);
  std::string binaryExtension = extPrefix.str() + ".bc";
  IREE::HAL::dumpDataToPath(path, baseName, suffix, binaryExtension,
                            StringRef(binaryData.data(), binaryData.size()));
}

static void fixupVisibility(llvm::Module &module,
                            const SetVector<llvm::Function *> &preserveFuncs) {
  for (auto &func : module) {
    if (preserveFuncs.contains(&func) || func.getName() == "iree_dll_main") {
      // Leave our library query function as public/external so that it is
      // exported from shared objects and available for linking in static
      // objects.
      continue;
    } else if (func.isDeclaration()) {
      // Declarations must have their original visibility/linkage; they most
      // often come from declared llvm builtin ops (llvm.memcpy/etc).
      continue;
    }
    func.setDSOLocal(true);
    func.setLinkage(llvm::GlobalValue::LinkageTypes::InternalLinkage);
  }
  for (auto &global : module.globals()) {
    global.setDSOLocal(true);
    global.setLinkage(llvm::GlobalValue::LinkageTypes::InternalLinkage);
  }
}

// Appends the |debugDatabase| to the end of |baseFile| and writes the footer
// so the runtime can find it.
static LogicalResult appendDebugDatabase(
    std::vector<int8_t> &baseFile, IREE::HAL::Artifact &debugFileArtifact) {
  auto debugFileOr = debugFileArtifact.read();
  if (!debugFileOr.has_value()) {
    return failure();
  }
  auto debugFile = std::move(debugFileOr).value();

  // NOTE: we align the sizes so that the files all start at nice offsets.
  auto baseFileSize = IREE::Util::align(baseFile.size(), 16);
  auto debugFileSize = IREE::Util::align(debugFile.size(), 16);

  // Matches iree_hal_system_executable_footer_t.
  struct Footer {
    uint8_t magic[8];  // IREEDBG\0
    uint32_t version;
    uint32_t flags;
    uint64_t libraryOffset;
    uint64_t librarySize;
    uint64_t debugOffset;
    uint64_t debugSize;
  } footer = {{0}};
  std::memcpy(footer.magic, "IREEDBG\0", sizeof(footer.magic));
  footer.version = 0;
  footer.librarySize = baseFile.size();
  footer.debugOffset = baseFileSize;
  footer.debugSize = debugFile.size();

  baseFile.resize(baseFileSize + debugFileSize + sizeof(footer));
  std::memcpy(baseFile.data() + baseFileSize, debugFile.data(),
              debugFile.size());
  std::memcpy(baseFile.data() + baseFileSize + debugFileSize, &footer,
              sizeof(footer));
  return success();
}

static constexpr char kQueryFunctionName[] =
    "iree_hal_executable_library_query";

}  // namespace

LogicalResult CoralNPUOptions::validate(MLIRContext *context) const {
  Location loc = context ? UnknownLoc::get(context) : UnknownLoc::get(nullptr);
  if (dtcmSizeKb <= 0) {
    return emitError(loc) << "coralnpu-dtcm-size-kb must be positive, got "
                          << dtcmSizeKb;
  }
  if (numVectorRegisters <= 0) {
    return emitError(loc)
           << "coralnpu-num-vector-registers must be positive, got "
           << numVectorRegisters;
  }
  if (tileVectorAlignment < 0) {
    return emitError(loc)
           << "coralnpu-tile-vector-alignment must be non-negative, got "
           << tileVectorAlignment;
  }
  if (tileUnrollAlignment < 0) {
    return emitError(loc)
           << "coralnpu-tile-unroll-alignment must be non-negative, got "
           << tileUnrollAlignment;
  }
  if (tileReductionAlignment <= 0) {
    return emitError(loc)
           << "coralnpu-tile-reduction-alignment must be positive, got "
           << tileReductionAlignment;
  }
  if (tileParallelAlignment < 0) {
    return emitError(loc)
           << "coralnpu-tile-parallel-alignment must be non-negative, got "
           << tileParallelAlignment;
  }
  return success();
}

CoralNPUTargetBackend::CoralNPUTargetBackend(const CoralNPUOptions &options)
    : options_(options) {
  IREE::HAL::LLVMCPUTargetCLOptions clOptions;

  clOptions.targetTriple = "riscv32";
  clOptions.targetABI = options.targetABI;
  clOptions.targetCPUFeatures = options.targetCPUFeatures;
  clOptions.linkEmbedded = options.linkEmbedded;
  clOptions.debugSymbols = options.debugSymbols;
  clOptions.embeddedLinkerPath = options.embeddedLinkerPath;

  defaultOptions_ = clOptions.getTargetOptions();
}

std::string CoralNPUTargetBackend::getLegacyDefaultDeviceID() const {
  return "coralnpu";
}

void CoralNPUTargetBackend::getDefaultExecutableTargets(
    MLIRContext *context, StringRef deviceID, DictionaryAttr deviceConfigAttr,
    SmallVectorImpl<IREE::HAL::ExecutableTargetAttr> &executableTargetAttrs)
    const {
  executableTargetAttrs.push_back(
      getExecutableTarget(context, defaultOptions_.target));
}

IREE::HAL::ExecutableTargetAttr CoralNPUTargetBackend::getExecutableTarget(
    MLIRContext *context, const IREE::HAL::LLVMTarget &target) const {
  Builder b(context);
  SmallVector<NamedAttribute> configItems;
  target.storeToConfigAttrs(context, configItems);
  configItems.emplace_back(
      b.getStringAttr(IREE::Encoding::kEncodingResolverAttrName),
      IREE::CPU::CPUEncodingResolverAttr::get(context, {}));

  std::string format;
  if (target.linkStatic) {
    format += "static";
  } else {
    llvm::Triple targetTriple(target.getTriple());
    if (target.getLinkEmbedded()) {
      format += "embedded-elf-";
    } else {
      format += "system-";
      switch (targetTriple.getObjectFormat()) {
        case llvm::Triple::ObjectFormatType::COFF:
          format += "dll-";
          break;
        case llvm::Triple::ObjectFormatType::ELF:
          format += "elf-";
          break;
        case llvm::Triple::ObjectFormatType::MachO:
          format += "dylib-";
          break;
        case llvm::Triple::ObjectFormatType::Wasm:
          format += "wasm-";
          break;
        default:
          format += "unknown-";
          break;
      }
    }
    format += getIreeArchNameForTargetTriple(targetTriple);
  }
  return b.getAttr<IREE::HAL::ExecutableTargetAttr>(
      b.getStringAttr("coralnpu"), b.getStringAttr(format),
      b.getDictionaryAttr(configItems));
}

void CoralNPUTargetBackend::getDependentDialects(
    DialectRegistry &registry) const {
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  registry.insert<IREE::Codegen::IREECodegenDialect, IREE::CPU::IREECPUDialect,
                  IREE::LinalgExt::IREELinalgExtDialect,
                  IREE::VectorExt::IREEVectorExtDialect,
                  mlir::transform::TransformDialect, pdl::PDLDialect,
                  pdl_interp::PDLInterpDialect>();
}

void CoralNPUTargetBackend::buildConfigurationPassPipeline(
    IREE::HAL::ExecutableTargetAttr targetAttr, OpPassManager &passManager) {
  CoralNPUTileSizeSelectionRegisterOptions registerOptions;
  registerOptions.numVectorRegisters = options_.numVectorRegisters;
  registerOptions.vectorAlignment = options_.tileVectorAlignment;
  registerOptions.unrollAlignment = options_.tileUnrollAlignment;
  registerOptions.reductionAlignment = options_.tileReductionAlignment;
  registerOptions.parallelAlignment = options_.tileParallelAlignment;

  OpPassManager &funcPassManager =
      passManager.nest<ModuleOp>().nest<func::FuncOp>();

  funcPassManager.addPass(
      createCoralNPUTileSizeSelectionRegisterPass(registerOptions));

  CoralNPUTileSizeSelectionDTCMOptions dtcmOptions;
  dtcmOptions.dtcmSizeKb = options_.dtcmSizeKb;

  funcPassManager.addPass(createCoralNPUTileSizeSelectionDTCMPass(dtcmOptions));

  funcPassManager.addPass(createCoralNPUTileSizeSelectionWorkgroupPass());

  buildLLVMCPUCodegenConfigurationPassPipeline(passManager);
}

void CoralNPUTargetBackend::buildTranslationPassPipeline(
    IREE::HAL::ExecutableTargetAttr targetAttr, OpPassManager &passManager) {
  buildLLVMCPUCodegenPassPipeline(passManager, /*enableAArch64SME=*/false);
}

void CoralNPUTargetBackend::buildLinkingPassPipeline(
    OpPassManager &passManager) {
  buildLLVMCPULinkingPassPipeline(passManager, "coralnpu");
}

std::optional<IREE::HAL::LLVMTarget> CoralNPUTargetBackend::getVariantTarget(
    IREE::HAL::ExecutableVariantOp variantOp) {
  auto configAttr = variantOp.getTarget().getConfiguration();
  return IREE::HAL::LLVMTarget::loadFromConfigAttr(
      variantOp.getLoc(), configAttr, defaultOptions_.target);
}

LogicalResult CoralNPUTargetBackend::serializeExecutable(
    const SerializationOptions &options,
    IREE::HAL::ExecutableVariantOp variantOp, OpBuilder &executableBuilder) {
  llvm::LLVMContext context;
  auto maybeTarget = getVariantTarget(variantOp);
  if (!maybeTarget) return failure();
  const IREE::HAL::LLVMTarget &target = *maybeTarget;
  LLVM_DEBUG(dbgs() << "CoralNPU SerializeExecutable:\n"
                    << "-----------------------------\n";
             target.print(dbgs()));

  auto libraryName =
      variantOp->getParentOfType<IREE::HAL::ExecutableOp>().getName().str();

  if (target.getLinkEmbedded() && target.linkStatic) {
    return variantOp.emitError()
           << "cannot embed ELF and produce static library simultaneously";
  }

  auto targetMachine = IREE::HAL::createTargetMachine(target);
  if (!targetMachine) {
    return mlir::emitError(variantOp.getLoc())
           << "failed to create target machine for target triple '"
           << target.getTriple() << "'";
  }

  const llvm::Triple &targetTriple = targetMachine->getTargetTriple();
  variantOp.getInnerModule()->setAttr(
      LLVM::LLVMDialect::getTargetTripleAttrName(),
      executableBuilder.getStringAttr(targetTriple.str()));

  auto llvmModule = mlir::translateModuleToLLVMIR(variantOp.getInnerModule(),
                                                  context, libraryName);
  if (!llvmModule) {
    return variantOp.emitError() << "failed to translate the MLIR LLVM "
                                    "dialect to the native llvm::Module";
  }

  for (auto &func : *llvmModule) {
    func.addFnAttr("frame-pointer", "all");
    func.addFnAttr("no-builtins");
    func.addFnAttr("hot");
  }

  IREE::HAL::LibraryBuilder::Mode libraryBuilderMode =
      target.debugSymbols
          ? IREE::HAL::LibraryBuilder::Mode::INCLUDE_REFLECTION_ATTRS
          : IREE::HAL::LibraryBuilder::Mode::NONE;
  IREE::HAL::LibraryBuilder libraryBuilder(
      llvmModule.get(), libraryBuilderMode,
      IREE::HAL::LibraryBuilder::Version::LATEST);

  switch (target.sanitizerKind) {
    case IREE::HAL::SanitizerKind::kNone: {
      libraryBuilder.setSanitizerKind(
          IREE::HAL::LibraryBuilder::SanitizerKind::NONE);
      break;
    }
    case IREE::HAL::SanitizerKind::kAddress: {
      libraryBuilder.setSanitizerKind(
          IREE::HAL::LibraryBuilder::SanitizerKind::ADDRESS);
      for (auto &function : llvmModule->getFunctionList()) {
        function.addFnAttr(llvm::Attribute::SanitizeAddress);
      }
    } break;
    case IREE::HAL::SanitizerKind::kThread: {
      libraryBuilder.setSanitizerKind(
          IREE::HAL::LibraryBuilder::SanitizerKind::THREAD);
      for (auto &function : llvmModule->getFunctionList()) {
        function.addFnAttr(llvm::Attribute::SanitizeThread);
      }
    } break;
  }

  auto importsAttrName =
      StringAttr::get(variantOp.getContext(), "hal.executable.imports");
  if (auto importsAttr = variantOp->getAttrOfType<ArrayAttr>(importsAttrName)) {
    for (auto importAttr : importsAttr.getAsValueRange<ArrayAttr>()) {
      auto nameAttr = cast<StringAttr>(importAttr[0]);
      auto weakAttr = cast<BoolAttr>(importAttr[1]);
      libraryBuilder.addImport(nameAttr.getValue(), weakAttr.getValue());
    }
    variantOp->removeAttr(importsAttrName);
  }

  auto align16 = llvm::Attribute::getWithAlignment(context, llvm::Align(16));
  for (auto exportOp :
       variantOp.getBlock().getOps<IREE::HAL::ExecutableExportOp>()) {
    auto *llvmFunc = llvmModule->getFunction(exportOp.getName());
    if (!llvmFunc) continue;
    llvmFunc->setLinkage(llvm::GlobalValue::LinkageTypes::InternalLinkage);
    llvmFunc->setDSOLocal(true);

    for (unsigned i = 0; i <= 2; ++i) {
      llvmFunc->addParamAttr(i, llvm::Attribute::NonNull);
      llvmFunc->addParamAttr(i, llvm::Attribute::NoAlias);
      llvmFunc->addParamAttr(i, align16);
    }

    IREE::HAL::LibraryBuilder::DispatchAttrs dispatchAttrs = {};

    dispatchAttrs.localMemorySize = exportOp.getWorkgroupLocalMemory()
                                        .value_or(APInt(64, 0))
                                        .getSExtValue();

    if (auto layoutAttr = exportOp.getLayout()) {
      dispatchAttrs.constantCount = layoutAttr.getConstants();
      dispatchAttrs.bindingCount = layoutAttr.getBindings().size();
    }

    if (auto workgroupSizeAttr = exportOp.getWorkgroupSize()) {
      auto workgroupSizeValues = workgroupSizeAttr->getValue();
      dispatchAttrs.workgroupSize[0] = static_cast<uint32_t>(
          cast<IntegerAttr>(workgroupSizeValues[0]).getInt());
      dispatchAttrs.workgroupSize[1] = static_cast<uint32_t>(
          cast<IntegerAttr>(workgroupSizeValues[1]).getInt());
      dispatchAttrs.workgroupSize[2] = static_cast<uint32_t>(
          cast<IntegerAttr>(workgroupSizeValues[2]).getInt());
    }

    IREE::HAL::LibraryBuilder::SourceLocation sourceLocation;
    if (options.debugLevel >= 1) {
      if (auto loc = findFirstFileLoc(exportOp.getLoc())) {
        sourceLocation = {"", loc->getFilename().str(), loc->getLine()};
      }
    }
    SmallVector<IREE::HAL::LibraryBuilder::SourceLocation> stageLocations;
    if (options.debugLevel >= 3) {
      if (auto locsAttr = exportOp.getSourceLocsAttr()) {
        for (auto locAttr : locsAttr.getValue()) {
          if (auto loc =
                  findFirstFileLoc(cast<LocationAttr>(locAttr.getValue()))) {
            stageLocations.push_back({
                locAttr.getName().str(),
                loc->getFilename().str(),
                loc->getLine(),
            });
          }
        }
      }
    }
    libraryBuilder.addExport(exportOp.getName(), std::move(sourceLocation),
                             std::move(stageLocations), /*tag=*/"",
                             dispatchAttrs, llvmFunc);
  }

  if (auto sourcesAttr = variantOp.getSourcesAttr()) {
    for (auto sourceAttr : sourcesAttr.getValue()) {
      if (auto resourceAttr = dyn_cast_if_present<DenseResourceElementsAttr>(
              sourceAttr.getValue())) {
        auto handle = resourceAttr.getRawHandle();
        SmallVector<char> rawData;
        llvm::append_range(rawData, handle.getBlob()->getData());
        libraryBuilder.addSourceFile(sourceAttr.getName(), std::move(rawData));
      }
    }
  }

  auto queryFunctionName = std::string(kQueryFunctionName);
  if (target.linkStatic) {
    queryFunctionName = libraryName + "_library_query";
  }
  auto *queryLibraryFunc = libraryBuilder.build(queryFunctionName);

  queryLibraryFunc->setDSOLocal(false);
  queryLibraryFunc->setVisibility(
      llvm::GlobalValue::VisibilityTypes::DefaultVisibility);
  queryLibraryFunc->setLinkage(
      llvm::GlobalValue::LinkageTypes::ExternalLinkage);
  queryLibraryFunc->setDLLStorageClass(
      llvm::GlobalValue::DLLStorageClassTypes::DLLExportStorageClass);

  std::unique_ptr<iree_compiler::IREE::HAL::LinkerTool> linkerTool;

  if (!target.linkStatic) {
    iree_compiler::IREE::HAL::LLVMTargetOptions targetOptions = defaultOptions_;
    targetOptions.target = target;

    if (target.getLinkEmbedded()) {
      linkerTool = createCoralNPULinkerTool(targetTriple, targetOptions);
    } else {
      linkerTool = iree_compiler::IREE::HAL::LinkerTool::getForTarget(
          targetTriple, targetOptions);
    }

    if (!linkerTool) {
      return mlir::emitError(variantOp.getLoc())
             << "failed to find a CoralNPU linker for target triple '"
             << targetTriple.str() << "'";
    }

    if (mlir::failed(linkerTool->configureModule(llvmModule.get(),
                                                 {queryLibraryFunc}))) {
      return variantOp.emitError()
             << "failed to configure LLVM module for CoralNPU linker";
    }
  }

  llvmModule->setDataLayout(targetMachine->createDataLayout());
  llvmModule->setTargetTriple(targetMachine->getTargetTriple());

  if (!options.dumpIntermediatesPath.empty()) {
    dumpLLVMModuleToPath(options.dumpIntermediatesPath, options.dumpBaseName,
                         variantOp.getName(), ".codegen", *llvmModule);
  }

  llvm::Linker moduleLinker(*llvmModule);

  if (failed(IREE::HAL::linkCmdlineBitcodeFiles(
          variantOp.getLoc(), moduleLinker, llvm::Linker::OverrideFromSrc,
          *targetMachine, context))) {
    return failure();
  }

  if (failed(IREE::HAL::linkBitcodeObjects(
          variantOp.getLoc(), moduleLinker, llvm::Linker::LinkOnlyNeeded,
          *targetMachine, variantOp.getObjectsAttr(), context))) {
    return failure();
  }

  if (failed(IREE::HAL::linkBitcodeModule(
          variantOp.getLoc(), moduleLinker, llvm::Linker::OverrideFromSrc,
          *targetMachine, "libdevice",
          IREE::HAL::loadDeviceBitcode(targetMachine.get(), context),
          [&](llvm::Module &module) {
            IREE::HAL::specializeDeviceModule(variantOp, module,
                                              *targetMachine);
          }))) {
    return mlir::emitError(variantOp.getLoc())
           << "failed linking in builtin library for target triple '"
           << targetTriple.str() << "'";
  }

  if (target.getLinkEmbedded()) {
    if (failed(IREE::HAL::linkBitcodeModule(
            variantOp.getLoc(), moduleLinker, llvm::Linker::OverrideFromSrc,
            *targetMachine, "libmusl",
            IREE::HAL::loadMuslBitcode(targetMachine.get(), context)))) {
      return mlir::emitError(variantOp.getLoc())
             << "failed linking in builtin library for target triple '"
             << targetTriple.str() << "'";
    }
  }

  if (target.linkUkernelBitcode) {
    if (hasUkernel(variantOp.getTarget().getConfiguration())) {
      llvm::Expected<std::unique_ptr<llvm::Module>> bitcode =
          IREE::HAL::loadUKernelBitcode(targetMachine.get(), context);
      if (!bitcode) {
        return mlir::emitError(variantOp.getLoc())
               << "failed to load ukernel bitcode: "
               << llvm::toString(bitcode.takeError());
      }

      if (bitcode.get()) {
        StringRef bitcodeName = bitcode.get()->getName();
        if (failed(IREE::HAL::linkBitcodeModule(
                variantOp.getLoc(), moduleLinker, llvm::Linker::LinkOnlyNeeded,
                *targetMachine, bitcodeName, std::move(bitcode), {}))) {
          return mlir::emitError(variantOp.getLoc())
                 << "failed linking in architecture-specific ukernel bitcode "
                    "for target triple '"
                 << targetTriple.str() << "'";
        }
      }
    }
  }

  auto *llvmIdent = llvmModule->getNamedMetadata("llvm.ident");
  if (llvmIdent) llvmIdent->clearOperands();

  if (!options.dumpIntermediatesPath.empty()) {
    dumpLLVMModuleToPath(options.dumpIntermediatesPath, options.dumpBaseName,
                         variantOp.getName(), ".linked", *llvmModule);
  }

  if (failed(IREE::HAL::runLLVMIRPasses(target, targetMachine.get(),
                                        llvmModule.get()))) {
    return variantOp.emitError()
           << "failed to run LLVM-IR opt passes for IREE::HAL::ExecutableOp "
              "targeting '"
           << targetTriple.str() << "'";
  }

  SetVector<llvm::Function *> preservedFuncs;
  preservedFuncs.insert(queryLibraryFunc);
  fixupVisibility(*llvmModule, preservedFuncs);

  if (!options.dumpIntermediatesPath.empty()) {
    dumpLLVMModuleToPath(options.dumpIntermediatesPath, options.dumpBaseName,
                         variantOp.getName(), ".optimized", *llvmModule);
  }

  SmallVector<IREE::HAL::Artifact> objectFiles;

  {
    std::string objectData;
    if (failed(IREE::HAL::runEmitObjFilePasses(
            targetMachine.get(), llvmModule.get(),
            llvm::CodeGenFileType::ObjectFile, &objectData))) {
      return variantOp.emitError()
             << "failed to compile LLVM-IR module to an object file";
    }
    if (!options.dumpIntermediatesPath.empty()) {
      IREE::HAL::dumpDataToPath(options.dumpIntermediatesPath,
                                options.dumpBaseName, variantOp.getName(), ".o",
                                objectData);
    }
    auto objectFile = IREE::HAL::Artifact::createTemporary(libraryName, "o");
    auto &os = objectFile.outputFile->os();
    os << objectData;
    os.flush();
    os.close();
    objectFiles.push_back(std::move(objectFile));
  }

  if (!options.dumpIntermediatesPath.empty()) {
    std::string asmData;
    if (failed(IREE::HAL::runEmitObjFilePasses(
            targetMachine.get(), llvmModule.get(),
            llvm::CodeGenFileType::AssemblyFile, &asmData))) {
      return variantOp.emitError()
             << "failed to compile LLVM-IR module to an assembly file";
    }
    IREE::HAL::dumpDataToPath(options.dumpIntermediatesPath,
                              options.dumpBaseName, variantOp.getName(), ".s",
                              asmData);
  }

  SmallVector<IREE::HAL::ExecutableObjectAttr> linkerObjectAttrs;
  IREE::HAL::ExecutableObjectAttr::filterObjects(variantOp.getObjectsAttr(),
                                                 {".o", ".obj", ".a", ".lib"},
                                                 linkerObjectAttrs);
  for (auto [index, attr] : llvm::enumerate(linkerObjectAttrs)) {
    auto objectAttr = cast<IREE::HAL::ExecutableObjectAttr>(attr);
    if (auto dataAttr = objectAttr.getData()) {
      objectFiles.push_back(IREE::HAL::Artifact::createTemporary(
          objectFiles.front().path + "_object_" + std::to_string(index),
          llvm::sys::path::extension(objectAttr.getPath())));
    } else {
      auto absolutePath = objectAttr.getAbsolutePath();
      if (failed(absolutePath)) {
        llvm::errs()
            << "ERROR: referenced object file not found on any path; use "
               "--iree-hal-executable-object-search-path= to add search "
               "paths: "
            << objectAttr << "\n";
        return failure();
      }
      objectFiles.push_back(IREE::HAL::Artifact::fromFile(*absolutePath));
    }
  }

  if (target.linkStatic) {
    return serializeStaticLibraryExecutable(options, target, variantOp,
                                            executableBuilder, libraryName,
                                            queryFunctionName, objectFiles);
  } else {
    return serializeDynamicLibraryExecutable(
        options, target, variantOp, executableBuilder, libraryName,
        targetTriple, objectFiles, linkerTool.get());
  }
}

LogicalResult CoralNPUTargetBackend::serializeStaticLibraryExecutable(
    const SerializationOptions &options, const IREE::HAL::LLVMTarget &target,
    IREE::HAL::ExecutableVariantOp variantOp, OpBuilder &executableBuilder,
    const std::string &libraryName, const std::string &queryFunctionName,
    const SmallVector<IREE::HAL::Artifact> &objectFiles) {
  if (objectFiles.size() != 1) {
    return variantOp.emitError() << "generating static libraries from "
                                    "multiple object files is not supported";
  }

  if (!IREE::HAL::outputStaticLibrary(libraryName, queryFunctionName,
                                      target.staticLibraryOutput,
                                      objectFiles[0].path)) {
    return variantOp.emitError() << "static library generation failed";
  }

  std::vector<uint8_t> libraryNameVector(libraryName.begin(),
                                         libraryName.end());
  libraryNameVector.push_back(0);  // NUL
  IREE::HAL::ExecutableBinaryOp::create(executableBuilder, variantOp.getLoc(),
                                        variantOp.getSymName(), "static",
                                        libraryNameVector);

  return success();
}

LogicalResult CoralNPUTargetBackend::serializeDynamicLibraryExecutable(
    const SerializationOptions &options, const IREE::HAL::LLVMTarget &target,
    IREE::HAL::ExecutableVariantOp variantOp, OpBuilder &executableBuilder,
    const std::string &libraryName, const llvm::Triple &targetTriple,
    const SmallVector<IREE::HAL::Artifact> &objectFiles,
    IREE::HAL::LinkerTool *linkerTool) {
  auto linkArtifactsOr =
      linkerTool->linkDynamicLibrary(libraryName, objectFiles);
  if (!linkArtifactsOr.has_value()) {
    return mlir::emitError(variantOp.getLoc())
           << "failed to link executable and generate target dylib (check "
              "above for more specific error messages)";
  }
  auto &linkArtifacts = linkArtifactsOr.value();
  if (defaultOptions_.keepLinkerArtifacts) {
    mlir::emitRemark(variantOp.getLoc())
        << "linker artifacts for " << variantOp.getName() << " preserved:\n"
        << "    " << linkArtifacts.libraryFile.path;
    linkArtifacts.keepAllFiles();
    for (auto &objectFile : objectFiles) {
      objectFile.keep();
    }
  }

  if (target.getLinkEmbedded()) {
    auto elfFile = linkArtifacts.libraryFile.read();
    if (!elfFile.has_value()) {
      return variantOp.emitError() << "failed to read back dylib temp file at "
                                   << linkArtifacts.libraryFile.path;
    }
    if (!options.dumpBinariesPath.empty()) {
      IREE::HAL::dumpDataToPath<int8_t>(options.dumpBinariesPath,
                                        options.dumpBaseName,
                                        variantOp.getName(), ".so", *elfFile);
    }
    auto bufferAttr = DenseIntElementsAttr::get(
        VectorType::get({static_cast<int64_t>(elfFile->size())},
                        IntegerType::get(executableBuilder.getContext(), 8)),
        std::move(elfFile.value()));

    auto binaryOp = IREE::HAL::ExecutableBinaryOp::create(
        executableBuilder, variantOp.getLoc(), variantOp.getSymName(),
        variantOp.getTarget().getFormat(), bufferAttr);
    binaryOp.setMimeTypeAttr(
        executableBuilder.getStringAttr("application/x-elf"));
  } else {
    const char *mimeType = nullptr;
    const char *extension = "";
    switch (targetTriple.getObjectFormat()) {
      case llvm::Triple::ObjectFormatType::COFF:
        mimeType = "application/x-msdownload";
        extension = ".dll";
        break;
      case llvm::Triple::ObjectFormatType::ELF:
        mimeType = "application/x-elf";
        extension = ".so";
        break;
      case llvm::Triple::ObjectFormatType::MachO:
        mimeType = "application/x-dylib";
        extension = ".dylib";
        break;
      case llvm::Triple::ObjectFormatType::Wasm:
        mimeType = "application/wasm";
        extension = ".wasm";
        break;
      default:
        mimeType = "application/octet-stream";
        break;
    }

    auto dylibFile = linkArtifacts.libraryFile.read();
    if (!dylibFile.has_value()) {
      return variantOp.emitError() << "failed to read back dylib temp file at "
                                   << linkArtifacts.libraryFile.path;
    }

    auto baseFile = std::move(dylibFile).value();
    if (linkArtifacts.debugFile.path.empty()) {
      // No debug database.
    } else {
      if (failed(appendDebugDatabase(baseFile, linkArtifacts.debugFile))) {
        return variantOp.emitError() << "failed to append debug database";
      }
    }

    if (!options.dumpBinariesPath.empty()) {
      IREE::HAL::dumpDataToPath<int8_t>(
          options.dumpBinariesPath, options.dumpBaseName, variantOp.getName(),
          extension, baseFile);
    }

    auto bufferAttr = DenseIntElementsAttr::get(
        VectorType::get({static_cast<int64_t>(baseFile.size())},
                        IntegerType::get(executableBuilder.getContext(), 8)),
        std::move(baseFile));

    auto binaryOp = IREE::HAL::ExecutableBinaryOp::create(
        executableBuilder, variantOp.getLoc(), variantOp.getSymName(),
        variantOp.getTarget().getFormat(), bufferAttr);
    if (mimeType) {
      binaryOp.setMimeTypeAttr(executableBuilder.getStringAttr(mimeType));
    }
  }

  return success();
}

}  // namespace mlir::coralnpu_compiler
