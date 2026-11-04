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

// The compiler metrics bundle of one executable (coralnpu.metrics.v1).

#ifndef METRICS_COMPILER_METRICSBUNDLE_H_
#define METRICS_COMPILER_METRICSBUNDLE_H_

#include "metrics/compiler/MetricsElf.h"

// LLVM:
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"

// Standard library:
#include <array>
#include <cstdint>
#include <optional>
#include <string>

namespace llvm {
class raw_ostream;
}  // namespace llvm

namespace mlir::coralnpu_compiler {

// Schema identifier of the metrics bundle. The JSON bundle has the keys
// below; a key is left out when its value is not known. Adding keys keeps
// the version; renaming or removing keys bumps it.
//   schema      kMetricsSchema
//   executable  the HAL executable name, e.g. main_dispatch_0
//   target      {triple, cpu_features, abi, vlen, float, zvt, dtcm_kb,
//               num_vector_registers}: MetricsTargetInfo
//   dispatches  [{name, ordinal, workgroup_size[], binding_count,
//               constant_count, ...}]: MetricsDispatchInfo; the entries of
//               the `coralnpu.metrics` attribute of the dispatch function
//               follow, sorted by key (MetricsAttr.h):
//               tile_grid  {op, loops[], workgroup[], dtcm[], vector[],
//                          zvt_grid[]} (RecordTileGridPass.h)
//               zvt        {candidate, lowered} (RecordTileGridPass.h,
//                          RecordZvtLoweringPass.h)
//   elf         {size_bytes, regions, sections[]}: MetricsElfInfo
//               (MetricsElf.h)
inline constexpr llvm::StringLiteral kMetricsSchema = "coralnpu.metrics.v1";

// Target facts derived from the executable target and the CoralNPU options.
struct MetricsTargetInfo {
  std::string triple;
  std::string cpuFeatures;
  std::string abi;
  // VLEN in bits (largest +zvl<N>b feature), if present.
  std::optional<int64_t> vlen;
  bool hasFloat = false;  // +f
  bool hasZvt = false;    // +zvtbase
  int dtcmSizeKb = 0;
  int numVectorRegisters = 0;
};

// Facts about one exported dispatch function.
struct MetricsDispatchInfo {
  std::string name;
  std::optional<int64_t> ordinal;
  std::optional<std::array<uint32_t, 3>> workgroupSize;
  int64_t bindingCount = 0;
  int64_t constantCount = 0;
  // Entries of the `coralnpu.metrics` attribute of the dispatch function.
  llvm::json::Object extra;
};

// Holds the compile-time metrics of one executable and writes them as a
// `coralnpu.metrics.v1` bundle (JSON) or as a human readable summary.
class MetricsBundle {
 public:
  // Creates an empty metrics bundle for |executableName|.
  explicit MetricsBundle(std::string executableName)
      : executableName(std::move(executableName)) {}

  // Returns the HAL executable name of this bundle.
  llvm::StringRef executable() const { return executableName; }

  // Sets the facts of the compilation target.
  void setTarget(MetricsTargetInfo info) { target = std::move(info); }

  // Adds the entry of one dispatch. The bundle keeps the order of the calls.
  void addDispatch(MetricsDispatchInfo info) {
    dispatches.push_back(std::move(info));
  }

  // Sets the size report of the linked ELF.
  void setElf(MetricsElfInfo info) { elf = std::move(info); }

  // Writes the bundle as `coralnpu.metrics.v1` JSON to |os|.
  void dumpJson(llvm::raw_ostream &os) const;

  // Writes a human-readable summary of the bundle to |os|.
  void dumpPretty(llvm::raw_ostream &os) const;

 private:
  std::string executableName;
  MetricsTargetInfo target;
  llvm::SmallVector<MetricsDispatchInfo> dispatches;
  std::optional<MetricsElfInfo> elf;
};

// Returns true if |key| is one of the fixed top-level keys of a dispatch entry
// in the metrics bundle ("name", "ordinal", "workgroup_size",
// "binding_count", "constant_count") that `coralnpu.metrics` attributes must
// not shadow.
bool isReservedDispatchKey(llvm::StringRef key);

}  // namespace mlir::coralnpu_compiler

#endif  // METRICS_COMPILER_METRICSBUNDLE_H_
