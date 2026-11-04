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

#include "metrics/compiler/MetricsTargetHooks.h"

#include "compiler/Target/Utils.h"
#include "metrics/compiler/MetricsAttr.h"
#include "metrics/compiler/MetricsBundle.h"
#include "metrics/compiler/MetricsElf.h"
#include "metrics/compiler/RecordTileGridPass.h"
#include "metrics/compiler/RecordZvtLoweringPass.h"

// IREE:
#include "compiler/plugins/target/LLVMCPU/LLVMTargetOptions.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"

// MLIR:
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/PassManager.h"

// LLVM:
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

// Standard library:
#include <array>
#include <optional>
#include <string>
#include <utility>

#define DEBUG_TYPE "coralnpu-metrics"

namespace mlir::coralnpu_compiler {

namespace {

namespace HAL = iree_compiler::IREE::HAL;

// Converts |attr| to JSON. Returns nullopt if |attr|, or a nested element, is
// not a supported kind or is an integer wider than 64 bits.
std::optional<llvm::json::Value> attrToJson(Attribute attr) {
  if (auto boolAttr = dyn_cast<BoolAttr>(attr)) return boolAttr.getValue();
  if (isa<UnitAttr>(attr)) return true;
  if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
    const APInt &value = intAttr.getValue();
    // llvm::json::Value holds integers of at most 64 bits.
    if (value.getBitWidth() > 64) return std::nullopt;
    if (intAttr.getType().isUnsignedInteger()) {
      return static_cast<uint64_t>(value.getZExtValue());
    }
    return static_cast<int64_t>(value.getSExtValue());
  }
  if (auto floatAttr = dyn_cast<FloatAttr>(attr)) {
    if (!floatAttr.getValue().isFinite()) return std::nullopt;
    return floatAttr.getValueAsDouble();
  }
  if (auto strAttr = dyn_cast<StringAttr>(attr)) return strAttr.str();
  if (auto arrayAttr = dyn_cast<ArrayAttr>(attr)) {
    llvm::json::Array array;
    for (Attribute element : arrayAttr) {
      std::optional<llvm::json::Value> value = attrToJson(element);
      if (!value) return std::nullopt;
      array.push_back(std::move(*value));
    }
    return llvm::json::Value(std::move(array));
  }
  if (auto dictAttr = dyn_cast<DictionaryAttr>(attr)) {
    llvm::json::Object object;
    for (NamedAttribute entry : dictAttr) {
      std::optional<llvm::json::Value> value = attrToJson(entry.getValue());
      if (!value) return std::nullopt;
      object[entry.getName().strref()] = std::move(*value);
    }
    return llvm::json::Value(std::move(object));
  }
  return std::nullopt;
}

// Copies the entries of the `coralnpu.metrics` attribute of function
// |funcName| in |innerModule| into |info.extra|. Skips the reserved keys and
// the unsupported values, which only a faulty metrics pass sets.
void readMetricsAttr(ModuleOp innerModule, llvm::StringRef funcName,
                     MetricsDispatchInfo &info) {
  Operation *funcOp = SymbolTable::lookupSymbolIn(innerModule, funcName);
  if (!funcOp) return;
  auto dictAttr = funcOp->getAttrOfType<DictionaryAttr>(kMetricsAttrName);
  if (!dictAttr) return;
  for (NamedAttribute entry : dictAttr) {
    llvm::StringRef key = entry.getName().strref();
    if (isReservedDispatchKey(key)) {
      LDBG() << "skipping the reserved metrics entry '" << key << "' of "
             << funcName;
      continue;
    }
    std::optional<llvm::json::Value> value = attrToJson(entry.getValue());
    if (!value) {
      LDBG() << "skipping the metrics entry '" << key << "' of " << funcName
             << ": unsupported value";
      continue;
    }
    info.extra[key] = std::move(*value);
  }
}

// Returns the target facts of |variantOp|: the triple, the CPU features, and
// the ABI of |target|, the target features queried via compiler/Target/Utils.h,
// and the facts of |config|.
MetricsTargetInfo makeTargetInfo(HAL::ExecutableVariantOp variantOp,
                                 const HAL::LLVMTarget &target,
                                 const MetricsTargetConfig &config) {
  MetricsTargetInfo targetInfo;
  targetInfo.triple = target.getTriple();
  targetInfo.cpuFeatures = target.getCpuFeatures();
  targetInfo.abi = target.llvmTargetOptions.MCOptions.ABIName;
  targetInfo.hasFloat = hasCoralNPUTargetFeature(variantOp, "+f");
  targetInfo.hasZvt = hasZvtTargetFeature(variantOp);
  if (FailureOr<int64_t> vlen =
          getVlenFromTargetFeatures(variantOp.getTarget().getConfiguration());
      succeeded(vlen)) {
    targetInfo.vlen = *vlen;
  }
  targetInfo.dtcmSizeKb = config.dtcmSizeKb;
  targetInfo.numVectorRegisters = config.numVectorRegisters;
  return targetInfo;
}

// Returns the entry of |exportOp|, with the `coralnpu.metrics` entries of its
// function in |innerModule|. |innerModule| is null for external variants.
MetricsDispatchInfo makeDispatchInfo(HAL::ExecutableExportOp exportOp,
                                     ModuleOp innerModule) {
  MetricsDispatchInfo dispatch;
  dispatch.name = exportOp.getName().str();
  // The ordinal is optional on the export.
  if (IntegerAttr ordinalAttr = exportOp.getOrdinalAttr()) {
    dispatch.ordinal = ordinalAttr.getInt();
  }
  // The workgroup size is optional on the export.
  if (auto workgroupSizeAttr = exportOp.getWorkgroupSize()) {
    std::array<uint32_t, 3> size = {1, 1, 1};
    // Keep at most 3 dimensions (x, y, z); missing dimensions stay 1.
    for (auto [index, value] :
         llvm::enumerate(workgroupSizeAttr->getAsValueRange<IntegerAttr>())) {
      if (index >= size.size()) break;
      size[index] = static_cast<uint32_t>(value.getZExtValue());
    }
    dispatch.workgroupSize = size;
  }
  // The layout, if set, gives the push constant and binding counts.
  if (auto layoutAttr = exportOp.getLayout()) {
    dispatch.constantCount = layoutAttr.getConstants();
    dispatch.bindingCount = layoutAttr.getBindings().size();
  }
  // External variants have no inner module, so no coralnpu.metrics entries.
  if (innerModule) readMetricsAttr(innerModule, dispatch.name, dispatch);
  return dispatch;
}

// The hooks that add the metrics passes and fill and write the metrics
// bundle of each executable.
class MetricsTargetHooks final : public CoralNPUTargetHooks {
 public:
  // Creates target hooks configured by |options| and |config|.
  MetricsTargetHooks(const MetricsOptions &options,
                     const MetricsTargetConfig &config)
      : options_(options), config_(config) {}

  // Adds the passes that record the per-dispatch facts of the configuration.
  void addConfigurationPasses(OpPassManager &funcPassManager) override {
    funcPassManager.addPass(createRecordTileGridPass());
  }

  // Adds the pass that records whether the matrix codegen used Zvt.
  void addPostMatrixCodegenPasses(OpPassManager &funcPassManager) override {
    funcPassManager.addPass(createRecordZvtLoweringPass());
  }

  // Builds and writes the metrics bundle of |variantOp| once it is serialized.
  void onExecutableSerialized(HAL::ExecutableVariantOp variantOp,
                              const HAL::LLVMTarget &target,
                              ArrayRef<uint8_t> linkedElf) override {
    MetricsBundle bundle(
        variantOp->getParentOfType<HAL::ExecutableOp>().getName().str());
    bundle.setTarget(makeTargetInfo(variantOp, target, config_));
    ModuleOp innerModule = variantOp.getInnerModule();
    // One entry per export, in the order of the exports.
    for (auto exportOp : variantOp.getExportOps()) {
      bundle.addDispatch(makeDispatchInfo(exportOp, innerModule));
    }
    if (!linkedElf.empty()) {
      // A bundle without the size report is still useful.
      if (llvm::Expected<MetricsElfInfo> elfOr = parseMetricsElf(linkedElf)) {
        bundle.setElf(std::move(*elfOr));
      } else {
        std::string error = llvm::toString(elfOr.takeError());
        LDBG() << "cannot parse the linked ELF of " << bundle.executable()
               << ": " << error;
      }
    }

    const bool json = options_.format == MetricsFormat::Json;
    std::string fileName =
        (bundle.executable() + ".metrics." + (json ? "json" : "txt")).str();
    writeReport(variantOp, options_.dir, fileName, [&](llvm::raw_ostream &os) {
      if (json) {
        bundle.dumpJson(os);
      } else {
        bundle.dumpPretty(os);
      }
    });
  }

 private:
  MetricsOptions options_;
  MetricsTargetConfig config_;
};

}  // namespace

void registerMetricsPasses() {
  registerRecordTileGridPass();
  registerRecordZvtLoweringPass();
}

std::shared_ptr<CoralNPUTargetHooks> createMetricsTargetHooks(
    const MetricsOptions &options, const MetricsTargetConfig &config) {
  if (!options.enabled()) return nullptr;
  return std::make_shared<MetricsTargetHooks>(options, config);
}

}  // namespace mlir::coralnpu_compiler
