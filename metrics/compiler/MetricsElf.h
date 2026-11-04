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

#ifndef METRICS_COMPILER_METRICSELF_H_
#define METRICS_COMPILER_METRICSELF_H_

// LLVM:
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

// Standard library:
#include <cstdint>
#include <optional>
#include <string>

namespace mlir::coralnpu_compiler {

// An allocated section of the linked ELF.
struct MetricsElfSection {
  std::string name;
  uint64_t address = 0;
  uint64_t size = 0;
  // "itcm", "dtcm" or "other" (by address).
  std::string region;
  // SHT_NOBITS (e.g. .bss, .heap, .stack): occupies memory, not file space.
  bool noBits = false;
};

// Usage of a memory region by the linked ELF.
struct MetricsElfRegion {
  uint64_t start = 0;
  uint64_t size = 0;
  // Bytes of the region used by the image, excluding the .heap and
  // .stack reservations (in DTCM, includes any alignment padding before
  // .heap so free() equals the .heap size).
  uint64_t used = 0;
  // Bytes reserved for the stack (DTCM only).
  uint64_t stack = 0;

  // Bytes left for the runtime (DTCM: binding tables, push constants and
  // local memory are allocated from what's left of the heap).
  uint64_t free() const {
    uint64_t taken = used + stack;
    return taken >= size ? 0 : size - taken;
  }
  // Percentage of |size| taken by |used| and |stack|; 0 if |size| is 0.
  double usedPercent() const {
    return size ? 100.0 * static_cast<double>(used + stack) / size : 0.0;
  }
};

// Size report of the linked ELF of an executable.
struct MetricsElfInfo {
  uint64_t fileSize = 0;
  llvm::SmallVector<MetricsElfSection> sections;
  // Present when the linker script defines __itcm_{start,size} and
  // __dtcm_{start,size} (the CoralNPU crt scripts do).
  std::optional<MetricsElfRegion> itcm;
  std::optional<MetricsElfRegion> dtcm;
};

// Parses the linked CoralNPU ELF |elfData| into a size report.
// It reads the linker script symbols __itcm_{start,size} and
// __dtcm_{start,size}, and the .stack and .heap sections. Keep in sync with
// the crt linker scripts (crt/BUILD.bazel, crt/CMakeLists.txt) and
// metrics/runtime/firmware_memory.c, which reads the same layout at run time.
llvm::Expected<MetricsElfInfo> parseMetricsElf(llvm::ArrayRef<uint8_t> elfData);

}  // namespace mlir::coralnpu_compiler

#endif  // METRICS_COMPILER_METRICSELF_H_
