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

#include "compiler/Transforms/Passes.h"

// IREE:
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtTypes.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"

// MLIR:
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

// LLVM:
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace mlir::iree_compiler;

namespace mlir::coralnpu_compiler {

#define GEN_PASS_DEF_CORALNPUDUMPAFFINITYEXECUTIONPROFILE
#include "compiler/Transforms/Passes.h.inc"

namespace {

struct TensorInfo {
  int64_t numElements = 1;
  Type elementType;
  bool isDynamic = false;
};

TensorInfo getTensorInfo(Type type) {
  TensorInfo info;
  info.elementType = type;

  if (auto dispatchTensorType =
          llvm::dyn_cast<IREE::TensorExt::DispatchTensorType>(type)) {
    if (!dispatchTensorType.hasStaticShape()) {
      info.isDynamic = true;
      return info;
    }
    info.numElements = dispatchTensorType.getNumElements();
    info.elementType = dispatchTensorType.getBoundElementType();
    return info;
  }

  if (auto shapedType = llvm::dyn_cast<ShapedType>(type)) {
    if (!shapedType.hasStaticShape()) {
      info.isDynamic = true;
      return info;
    }
    info.numElements = shapedType.getNumElements();
    info.elementType = shapedType.getElementType();
    return info;
  }

  return info;
}

static int64_t getElementTypeBitWidth(Type elType) {
  if (elType.isIntOrFloat() || elType.isIndex()) {
    return elType.getIntOrFloatBitWidth();
  }

  return 32;  // Default fallback
}

struct ExecutableFuncInfo {
  int64_t opCount = 0;
  int64_t elementCount = 0;
  int64_t logicalDataSize = 0;
  bool isDynamic = false;
};

ExecutableFuncInfo analyzeFunc(mlir::FunctionOpInterface funcOp) {
  ExecutableFuncInfo info;
  // Walk the body to avoid counting funcOp itself.
  funcOp.getFunctionBody().walk([&](Operation *op) { ++info.opCount; });

  funcOp.walk([&](IREE::Stream::BindingSubspanOp subspanOp) {
    Type resultType = subspanOp.getResult().getType();
    TensorInfo tensorInfo = getTensorInfo(resultType);

    if (tensorInfo.isDynamic) {
      info.isDynamic = true;
      return;
    }

    info.elementCount += tensorInfo.numElements;

    int64_t totalBits =
        tensorInfo.numElements * getElementTypeBitWidth(tensorInfo.elementType);
    info.logicalDataSize += llvm::divideCeil(totalBits, 8);
  });

  return info;
}

std::string resolveDeviceNameImpl(Attribute affinity, ModuleOp moduleOp) {
  if (!affinity) return "Default";

  auto deviceAffinity = llvm::dyn_cast<IREE::HAL::DeviceAffinityAttr>(affinity);
  if (!deviceAffinity) {
    std::string str;
    llvm::raw_string_ostream ss(str);
    affinity.print(ss);
    return str;
  }

  SymbolRefAttr deviceSymbol = deviceAffinity.getDevice();
  StringRef deviceName = deviceSymbol.getRootReference().getValue();

  auto globalOp = SymbolTable::lookupNearestSymbolFrom<IREE::Util::GlobalOp>(
      moduleOp, deviceSymbol);
  if (!globalOp) {
    return deviceName.str();
  }

  auto valueAttr = globalOp.getInitialValueAttr();
  if (!valueAttr) {
    return deviceName.str();
  }

  auto getTargetName = [](IREE::HAL::DeviceTargetAttr target) -> std::string {
    return target.getDeviceID().str();
  };

  if (auto targetAttr =
          llvm::dyn_cast<IREE::HAL::DeviceTargetAttr>(valueAttr)) {
    return getTargetName(targetAttr);
  }

  if (auto selectAttr =
          llvm::dyn_cast<IREE::HAL::DeviceSelectAttr>(valueAttr)) {
    std::string selectStr;
    llvm::raw_string_ostream ss(selectStr);
    bool first = true;
    for (auto devAttr : selectAttr.getDevices()) {
      if (!first) ss << "/";
      first = false;
      if (auto target = llvm::dyn_cast<IREE::HAL::DeviceTargetAttr>(devAttr)) {
        ss << getTargetName(target);
      } else {
        devAttr.print(ss);
      }
    }
    return selectStr;
  }

  return deviceName.str();
}

struct UsageInfo {
  llvm::DenseMap<Operation *, ExecutableFuncInfo> dispatchFuncInfos;
  llvm::DenseMap<Attribute, std::string> deviceNames;
  SmallVector<IREE::Stream::CmdExecuteOp> executeOps;

  std::string getDeviceName(Attribute affinity) const {
    if (!affinity) return "Default";
    auto it = deviceNames.find(affinity);
    if (it != deviceNames.end()) return it->second;
    return "Unknown";
  }

  void resolveDeviceName(Attribute affinity, ModuleOp moduleOp) {
    if (!affinity) return;
    if (llvm::is_contained(deviceNames, affinity)) return;
    deviceNames[affinity] = resolveDeviceNameImpl(affinity, moduleOp);
  }

  void analyze(mlir::ModuleOp moduleOp) {
    SymbolTable symbolTable(moduleOp);
    for (auto funcOp : moduleOp.getOps<mlir::FunctionOpInterface>()) {
      funcOp.walk(
          [&](IREE::Stream::CmdExecuteOp op) { executeOps.push_back(op); });
    }

    llvm::DenseMap<Operation *, ExecutableFuncInfo> funcInfos;
    for (auto executeOp : executeOps) {
      executeOp.walk([&](Operation *op) {
        if (auto affinity = IREE::Stream::AffinityAttr::lookup(op)) {
          resolveDeviceName(affinity, moduleOp);
        }

        if (auto dispatchOp = llvm::dyn_cast<IREE::Stream::CmdDispatchOp>(op)) {
          dispatchOp.forEachEntryPointAttr([&](SymbolRefAttr entryPointAttr) {
            auto exportOp = cast<IREE::Stream::ExecutableExportOp>(
                symbolTable.lookupSymbolIn(moduleOp, entryPointAttr));
            assert(exportOp && "missing executable/export");
            auto funcOp = exportOp.lookupFunctionRef();
            assert(funcOp && "missing exported function");

            auto it = funcInfos.find(funcOp);
            if (it == funcInfos.end()) {
              it = funcInfos.insert({funcOp, analyzeFunc(funcOp)}).first;
            }
            dispatchFuncInfos[dispatchOp.getOperation()] = it->second;
          });
        }
      });
    }
  }
};

struct AffinityStats {
  /// Total number of stream fill operations under this affinity.
  size_t fillCount = 0;
  /// Total number of stream copy operations under this affinity.
  size_t copyCount = 0;
  /// Total number of stream dispatches under this affinity.
  size_t dispatchCount = 0;

  /// Number of dispatches with statically known element/tensor shapes.
  size_t staticDispatchCount = 0;

  /// Whether at least one dispatch under this affinity has dynamic tensor
  /// shapes.
  bool hasDynamicDispatch = false;

  /// Total logical data size (in bytes) across all static dispatches.
  int64_t staticDispatchDataSize = 0;
  /// Total number of tensor elements across all static dispatches.
  int64_t staticDispatchElements = 0;
  /// Estimated work metric (logical data size * op count) for static
  /// dispatches.
  int64_t staticWorkBytes = 0;
  /// Estimated work metric (element count * op count) for static dispatches.
  int64_t staticWorkElements = 0;
};

static void analyzeDispatch(IREE::Stream::CmdDispatchOp dispatchOp,
                            AffinityStats &affStats,
                            const ExecutableFuncInfo &funcInfo) {
  if (funcInfo.isDynamic) {
    affStats.hasDynamicDispatch = true;
  } else {
    affStats.staticDispatchDataSize += funcInfo.logicalDataSize;
    affStats.staticDispatchElements += funcInfo.elementCount;
    affStats.staticWorkBytes += funcInfo.logicalDataSize * funcInfo.opCount;
    affStats.staticWorkElements += funcInfo.elementCount * funcInfo.opCount;
    ++affStats.staticDispatchCount;
  }
}

static void analyzeStatistics(
    const UsageInfo &usageInfo,
    llvm::MapVector<Attribute, AffinityStats> &affinityStats) {
  for (auto executeOp : usageInfo.executeOps) {
    executeOp.walk([&](Operation *op) {
      Attribute affinity = IREE::Stream::AffinityAttr::lookup(op);
      auto &affStats = affinityStats[affinity];
      TypeSwitch<Operation *>(op)
          .Case<IREE::Stream::CmdFillOp>([&](auto op) { ++affStats.fillCount; })
          .Case<IREE::Stream::CmdCopyOp>([&](auto op) { ++affStats.copyCount; })
          .Case<IREE::Stream::CmdDispatchOp>([&](auto op) {
            ++affStats.dispatchCount;
            auto it = usageInfo.dispatchFuncInfos.find(op);
            if (it != usageInfo.dispatchFuncInfos.end()) {
              analyzeDispatch(op, affStats, it->second);
            }
          });
    });
  }
}

std::unique_ptr<llvm::raw_fd_ostream> openOutputFile(StringRef filePath) {
  if (filePath.empty()) {
    return std::make_unique<llvm::raw_fd_ostream>(2, false);  // stderr
  }

  if (filePath == "-") {
    return std::make_unique<llvm::raw_fd_ostream>(1, false);  // stdout
  }

  std::error_code ec;
  auto result = std::make_unique<llvm::raw_fd_ostream>(
      filePath, ec, llvm::sys::fs::OF_TextWithCRLF);
  if (!ec) return result;
  llvm::errs() << "Error opening dump file '" << filePath
               << "': " << ec.message() << "\n";
  return std::make_unique<llvm::raw_fd_ostream>(2,
                                                false);  // Fallback to stderr
}

void printReportPretty(const UsageInfo &usageInfo,
                       const llvm::MapVector<Attribute, AffinityStats> &stats,
                       llvm::raw_ostream &os) {
  os << "======================================================================"
        "==\n";
  os << "Execution Profile by Affinity:\n";
  os << "======================================================================"
        "==\n";

  int64_t totalDispatchCount = 0;
  int64_t totalStaticWorkBytes = 0;
  int64_t totalStaticWorkElements = 0;
  for (const auto &[_, affStats] : stats) {
    totalDispatchCount += affStats.dispatchCount;
    totalStaticWorkBytes += affStats.staticWorkBytes;
    totalStaticWorkElements += affStats.staticWorkElements;
  }

  if (totalDispatchCount == 0) {
    os << "  No dispatches\n";
    os << "===================================================================="
          "====\n";
    return;
  }

  for (const auto &[affinity, affStats] : stats) {
    os << "  Affinity: "
       << (affinity ? usageInfo.getDeviceName(affinity) : "Default (CPU)")
       << "\n";

    double dispatchPct =
        (affStats.dispatchCount / (double)totalDispatchCount) * 100.0;
    os << llvm::formatv("    Dispatches: {0} ({1:F1}%)\n",
                        affStats.dispatchCount, dispatchPct);

    double staticPct =
        affStats.dispatchCount > 0
            ? (affStats.staticDispatchCount / (double)affStats.dispatchCount) *
                  100.0
            : 0.0;
    double dynamicPct =
        affStats.dispatchCount > 0
            ? ((affStats.dispatchCount - affStats.staticDispatchCount) /
               (double)affStats.dispatchCount) *
                  100.0
            : 0.0;
    os << llvm::formatv("      Static:    {0} ({1:F1}%)\n",
                        affStats.staticDispatchCount, staticPct);
    os << llvm::formatv("      Dynamic:   {0} ({1:F1}%)\n",
                        affStats.dispatchCount - affStats.staticDispatchCount,
                        dynamicPct);

    os << "    Estimated Static Data Size: ";
    os << llvm::formatv("{0:F2} MB",
                        affStats.staticDispatchDataSize / (1024.0f * 1024.0f));
    if (affStats.hasDynamicDispatch) {
      os << llvm::formatv(
          " (incomplete, {0} dynamic)",
          affStats.dispatchCount - affStats.staticDispatchCount);
    }
    os << "\n";

    os << "    Estimated Static Elements: ";
    os << llvm::formatv("{0:F2} M",
                        affStats.staticDispatchElements / 1000000.0f);
    if (affStats.staticDispatchElements > 0) {
      double bytesPerElement = (double)affStats.staticDispatchDataSize /
                               affStats.staticDispatchElements;
      os << llvm::formatv(" ({0:F2} bytes/element)", bytesPerElement);
    }
    if (affStats.hasDynamicDispatch) {
      os << llvm::formatv(
          " (incomplete, {0} dynamic)",
          affStats.dispatchCount - affStats.staticDispatchCount);
    }
    os << "\n";

    double workBytesPct =
        totalStaticWorkBytes > 0
            ? (affStats.staticWorkBytes / (double)totalStaticWorkBytes) * 100.0
            : 0.0;
    os << "    Estimated Static Work (Op-Bytes): ";
    os << llvm::formatv("{0:F2} M ({1:F1}%)",
                        affStats.staticWorkBytes / 1000000.0f, workBytesPct);
    if (affStats.hasDynamicDispatch) {
      os << " (incomplete)";
    }
    os << "\n";

    double workElemsPct =
        totalStaticWorkElements > 0
            ? (affStats.staticWorkElements / (double)totalStaticWorkElements) *
                  100.0
            : 0.0;
    os << "    Estimated Static Work (Op-Elems): ";
    os << llvm::formatv("{0:F2} M ({1:F1}%)",
                        affStats.staticWorkElements / 1000000.0f, workElemsPct);
    if (affStats.hasDynamicDispatch) {
      os << " (incomplete)";
    }
    os << "\n";
    os << llvm::formatv("    Fills:  {0}\n", affStats.fillCount);
    os << llvm::formatv("    Copies: {0}\n", affStats.copyCount);
  }
  os << "======================================================================"
        "==\n";
}

void printReportCSV(const UsageInfo &usageInfo,
                    const llvm::MapVector<Attribute, AffinityStats> &stats,
                    llvm::raw_ostream &os) {
  os << "\"Affinity\",\"Dispatches\",\"Static\",\"Dynamic\",\"Estimated Static "
        "Data Size (Bytes)\",\"Estimated Static Elements\",\"Estimated Static "
        "Work (Op-Bytes)\",\"Estimated Static Work "
        "(Op-Elems)\",\"Fills\",\"Copies\"";
  os << "\n";
  for (auto &it : stats) {
    Attribute affinity = it.first;
    const auto &affStats = it.second;

    std::string affinityStr = usageInfo.getDeviceName(affinity);
    os << llvm::formatv(
        R"("{0}",{1},{2},{3},{4},{5},{6},{7},{8},{9})", affinityStr,
        affStats.dispatchCount, affStats.staticDispatchCount,
        affStats.dispatchCount - affStats.staticDispatchCount,
        affStats.staticDispatchDataSize, affStats.staticDispatchElements,
        affStats.staticWorkBytes, affStats.staticWorkElements,
        affStats.fillCount, affStats.copyCount);
    os << "\n";
  }
}

void printReportJSON(const UsageInfo &usageInfo,
                     const llvm::MapVector<Attribute, AffinityStats> &stats,
                     llvm::raw_ostream &os) {
  os << "{\n";
  os << "  \"affinities\": {\n";
  bool firstAffinity = true;
  for (auto &it : stats) {
    if (!firstAffinity) {
      os << ",\n";
    }
    firstAffinity = false;

    Attribute affinity = it.first;
    const auto &affStats = it.second;

    os << "    \"" << usageInfo.getDeviceName(affinity) << "\": {\n";
    os << llvm::formatv("      \"dispatch-count\": {0},\n",
                        affStats.dispatchCount);
    os << llvm::formatv("      \"static-dispatch-count\": {0},\n",
                        affStats.staticDispatchCount);
    os << llvm::formatv("      \"dynamic-dispatch-count\": {0},\n",
                        affStats.dispatchCount - affStats.staticDispatchCount);
    os << llvm::formatv("      \"static-data-size-bytes\": {0},\n",
                        affStats.staticDispatchDataSize);
    os << llvm::formatv("      \"has-dynamic-data-size\": {0},\n",
                        affStats.hasDynamicDispatch ? "true" : "false");
    os << llvm::formatv("      \"static-elements-count\": {0},\n",
                        affStats.staticDispatchElements);
    os << llvm::formatv("      \"has-dynamic-elements\": {0},\n",
                        affStats.hasDynamicDispatch ? "true" : "false");
    os << llvm::formatv("      \"static-work-bytes\": {0},\n",
                        affStats.staticWorkBytes);
    os << llvm::formatv("      \"static-work-elements\": {0},\n",
                        affStats.staticWorkElements);
    os << llvm::formatv("      \"fill-count\": {0},\n", affStats.fillCount);
    os << llvm::formatv("      \"copy-count\": {0}\n", affStats.copyCount);
    os << "    }";
  }
  os << "\n  }\n";
  os << "}\n";
}

struct CoralNPUDumpAffinityExecutionProfilePass
    : public impl::CoralNPUDumpAffinityExecutionProfileBase<
          CoralNPUDumpAffinityExecutionProfilePass> {
  using impl::CoralNPUDumpAffinityExecutionProfileBase<
      CoralNPUDumpAffinityExecutionProfilePass>::
      CoralNPUDumpAffinityExecutionProfileBase;

  void runOnOperation() override {
    if (outputFormat == DumpOutputFormat::None) return;

    ModuleOp moduleOp = getOperation();
    UsageInfo usageInfo;
    usageInfo.analyze(moduleOp);

    llvm::MapVector<Attribute, AffinityStats> stats;
    analyzeStatistics(usageInfo, stats);

    auto os = openOutputFile(outputFile);
    if (!os) {
      return;
    }

    switch (outputFormat) {
      case DumpOutputFormat::Pretty:
      case DumpOutputFormat::Verbose:
        printReportPretty(usageInfo, stats, *os);
        break;
      case DumpOutputFormat::CSV:
        printReportCSV(usageInfo, stats, *os);
        break;
      case DumpOutputFormat::JSON:
        printReportJSON(usageInfo, stats, *os);
        break;
      default:
        break;
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>>
createCoralNPUDumpAffinityExecutionProfilePass() {
  return std::make_unique<CoralNPUDumpAffinityExecutionProfilePass>();
}

std::unique_ptr<OperationPass<ModuleOp>>
createCoralNPUDumpAffinityExecutionProfilePass(
    CoralNPUDumpAffinityExecutionProfileOptions options) {
  return std::make_unique<CoralNPUDumpAffinityExecutionProfilePass>(
      std::move(options));
}

}  // namespace mlir::coralnpu_compiler
