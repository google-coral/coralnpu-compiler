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

#include "metrics/compiler/MetricsBundle.h"

// LLVM:
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

// Standard library:
#include <algorithm>
#include <cinttypes>

namespace mlir::coralnpu_compiler {

namespace {

// Returns the keys of |object| in sorted order, for a stable output.
llvm::SmallVector<llvm::StringRef> sortedKeys(
    const llvm::json::Object &object) {
  llvm::SmallVector<llvm::StringRef> keys;
  for (const auto &entry : object) keys.push_back(entry.first);
  llvm::sort(keys);
  return keys;
}

// Returns "Yes" or "No", for the text report.
const char *yesNo(bool value) { return value ? "Yes" : "No"; }

// Writes |region| as the JSON object |name|. |withStack| adds the stack
// reservation (DTCM only).
void dumpRegionJson(llvm::json::OStream &jsonWriter, llvm::StringRef name,
                    const MetricsElfRegion &region, bool withStack) {
  jsonWriter.attributeObject(name, [&] {
    jsonWriter.attribute("start", region.start);
    jsonWriter.attribute("size", region.size);
    jsonWriter.attribute("used", region.used);
    if (withStack) jsonWriter.attribute("stack", region.stack);
    jsonWriter.attribute("free", region.free());
  });
}

// Writes the size report |elf| as the "elf" JSON object.
void dumpElfJson(llvm::json::OStream &jsonWriter, const MetricsElfInfo &elf) {
  jsonWriter.attributeObject("elf", [&] {
    jsonWriter.attribute("size_bytes", elf.fileSize);
    jsonWriter.attributeObject("regions", [&] {
      if (elf.itcm)
        dumpRegionJson(jsonWriter, "itcm", *elf.itcm, /*withStack=*/false);
      if (elf.dtcm)
        dumpRegionJson(jsonWriter, "dtcm", *elf.dtcm, /*withStack=*/true);
    });
    jsonWriter.attributeArray("sections", [&] {
      for (const MetricsElfSection &section : elf.sections) {
        jsonWriter.object([&] {
          jsonWriter.attribute("name", section.name);
          jsonWriter.attribute("region", section.region);
          jsonWriter.attribute("address", section.address);
          jsonWriter.attribute("size", section.size);
          jsonWriter.attribute("type", section.noBits ? "nobits" : "progbits");
        });
      }
    });
  });
}

// Writes |region| as one text line. |withStack| adds the stack reservation
// (DTCM only).
void dumpRegionPretty(llvm::raw_ostream &os, llvm::StringRef name,
                      const MetricsElfRegion &region, bool withStack) {
  os << "    " << llvm::left_justify(name, 6)
     << llvm::format_decimal(region.used + region.stack, 8) << " / "
     << llvm::format_decimal(region.size, 8) << " bytes ("
     << llvm::format("%5.1f", region.usedPercent()) << "%)";
  if (withStack) os << ", stack " << region.stack;
  os << ", free " << region.free() << "\n";
}

// Writes the size report |elf| as text: the regions, then the sections.
void dumpElfPretty(llvm::raw_ostream &os, const MetricsElfInfo &elf) {
  os << "ELF:\n";
  os << "  File Size (bytes): " << elf.fileSize << "\n";
  if (elf.itcm || elf.dtcm) os << "  Regions:\n";
  if (elf.itcm) dumpRegionPretty(os, "ITCM", *elf.itcm, /*withStack=*/false);
  if (elf.dtcm) dumpRegionPretty(os, "DTCM", *elf.dtcm, /*withStack=*/true);
  os << "  Sections:\n";
  os << "    " << llvm::left_justify("Name", 16)
     << llvm::left_justify("Region", 8) << llvm::left_justify("Address", 12)
     << llvm::right_justify("Size", 8) << "  Type\n";
  for (const MetricsElfSection &section : elf.sections) {
    os << "    " << llvm::left_justify(section.name, 16)
       << llvm::left_justify(section.region, 8)
       << llvm::format("0x%08" PRIx64 "  ", section.address)
       << llvm::format_decimal(section.size, 8) << "  "
       << (section.noBits ? "nobits" : "progbits") << "\n";
  }
}

}  // namespace

bool isReservedDispatchKey(llvm::StringRef key) {
  constexpr llvm::StringLiteral kReservedDispatchKeys[] = {
      "name", "ordinal", "workgroup_size", "binding_count", "constant_count"};
  return llvm::is_contained(kReservedDispatchKeys, key);
}

void MetricsBundle::dumpJson(llvm::raw_ostream &os) const {
  llvm::json::OStream jsonWriter(os, /*IndentSize=*/2);
  jsonWriter.object([&] {
    jsonWriter.attribute("schema", kMetricsSchema);
    jsonWriter.attribute("executable", executableName);
    jsonWriter.attributeObject("target", [&] {
      jsonWriter.attribute("triple", target.triple);
      jsonWriter.attribute("cpu_features", target.cpuFeatures);
      jsonWriter.attribute("abi", target.abi);
      if (target.vlen) jsonWriter.attribute("vlen", *target.vlen);
      jsonWriter.attribute("float", target.hasFloat);
      jsonWriter.attribute("zvt", target.hasZvt);
      jsonWriter.attribute("dtcm_kb", target.dtcmSizeKb);
      jsonWriter.attribute("num_vector_registers", target.numVectorRegisters);
    });
    jsonWriter.attributeArray("dispatches", [&] {
      for (const MetricsDispatchInfo &dispatch : dispatches) {
        jsonWriter.object([&] {
          jsonWriter.attribute("name", dispatch.name);
          if (dispatch.ordinal)
            jsonWriter.attribute("ordinal", *dispatch.ordinal);
          if (dispatch.workgroupSize) {
            jsonWriter.attributeArray("workgroup_size", [&] {
              for (uint32_t size : *dispatch.workgroupSize)
                jsonWriter.value(size);
            });
          }
          jsonWriter.attribute("binding_count", dispatch.bindingCount);
          jsonWriter.attribute("constant_count", dispatch.constantCount);
          for (llvm::StringRef key : sortedKeys(dispatch.extra)) {
            jsonWriter.attribute(key, *dispatch.extra.get(key));
          }
        });
      }
    });
    if (elf) dumpElfJson(jsonWriter, *elf);
  });
  os << "\n";
}

void MetricsBundle::dumpPretty(llvm::raw_ostream &os) const {
  constexpr llvm::StringLiteral kRule =
      "========================================================================"
      "\n";
  os << kRule << "CoralNPU Metrics: " << executableName << "\n" << kRule;
  os << "Target:\n";
  os << "  Triple: " << target.triple << "\n";
  os << "  CPU Features: " << target.cpuFeatures << "\n";
  os << "  ABI: " << target.abi << "\n";
  os << "  VLEN: ";
  if (target.vlen) {
    os << *target.vlen << "\n";
  } else {
    os << "n/a\n";
  }
  os << "  Float: " << yesNo(target.hasFloat)
     << ", Zvt: " << yesNo(target.hasZvt) << "\n";
  os << "  DTCM (KB): " << target.dtcmSizeKb
     << ", Vector Registers: " << target.numVectorRegisters << "\n";
  os << "Dispatches: " << dispatches.size() << "\n";
  for (const MetricsDispatchInfo &dispatch : dispatches) {
    os << "  Dispatch: " << dispatch.name << "\n";
    if (dispatch.ordinal) os << "    Ordinal: " << *dispatch.ordinal << "\n";
    if (dispatch.workgroupSize) {
      const auto &size = *dispatch.workgroupSize;
      os << "    Workgroup Size: " << size[0] << "x" << size[1] << "x"
         << size[2] << "\n";
    }
    os << "    Bindings: " << dispatch.bindingCount
       << ", Constants: " << dispatch.constantCount << "\n";
    for (llvm::StringRef key : sortedKeys(dispatch.extra)) {
      os << "    " << key << ": " << *dispatch.extra.get(key) << "\n";
    }
  }
  if (elf) dumpElfPretty(os, *elf);
}

}  // namespace mlir::coralnpu_compiler
