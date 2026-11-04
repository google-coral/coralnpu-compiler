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

#include "metrics/compiler/MetricsElf.h"

// LLVM:
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Object/ELF.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBufferRef.h"

// Standard library:
#include <algorithm>
#include <memory>
#include <utility>

namespace mlir::coralnpu_compiler {

namespace {

// Returns whether [address, address + size) lies within |region|.
bool regionContains(const std::optional<MetricsElfRegion> &region,
                    uint64_t address, uint64_t size) {
  return region && address >= region->start &&
         address - region->start <= region->size &&
         size <= region->size - (address - region->start);
}

// Returns the number of bytes that [aStart, aStart + aSize) shares with
// [bStart, bStart + bSize).
uint64_t rangeOverlap(uint64_t aStart, uint64_t aSize, uint64_t bStart,
                      uint64_t bSize) {
  const uint64_t begin = std::max(aStart, bStart);
  const uint64_t end = std::min(aStart + aSize, bStart + bSize);
  return end > begin ? end - begin : 0;
}

}  // namespace

llvm::Expected<MetricsElfInfo> parseMetricsElf(
    llvm::ArrayRef<uint8_t> elfData) {
  llvm::MemoryBufferRef buffer(
      llvm::StringRef(reinterpret_cast<const char *>(elfData.data()),
                      elfData.size()),
      "coralnpu-elf");
  llvm::Expected<std::unique_ptr<llvm::object::ObjectFile>> objectOr =
      llvm::object::ObjectFile::createELFObjectFile(buffer);
  if (!objectOr) return objectOr.takeError();
  auto *elfObject =
      llvm::dyn_cast<llvm::object::ELF32LEObjectFile>(objectOr->get());
  if (!elfObject) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "not a 32-bit little-endian ELF file");
  }

  MetricsElfInfo info;
  info.fileSize = elfData.size();

  // Region bounds come from defined linker script symbols.
  llvm::StringMap<uint64_t> symbols;
  for (const llvm::object::SymbolRef &symbol : elfObject->symbols()) {
    llvm::Expected<uint32_t> flags = symbol.getFlags();
    if (!flags) {
      llvm::consumeError(flags.takeError());
      continue;
    }
    if (*flags & llvm::object::SymbolRef::SF_Undefined) continue;
    llvm::Expected<llvm::StringRef> name = symbol.getName();
    if (!name) {
      llvm::consumeError(name.takeError());
      continue;
    }
    // Keep only the linker script symbols (__itcm_start, __dtcm_size, ...).
    if (!name->starts_with("__")) continue;
    llvm::Expected<uint64_t> value = symbol.getValue();
    if (!value) {
      llvm::consumeError(value.takeError());
      continue;
    }
    symbols[*name] = *value;
  }
  auto lookupRegion =
      [&](llvm::StringRef startSym,
          llvm::StringRef sizeSym) -> std::optional<MetricsElfRegion> {
    auto start = symbols.find(startSym);
    auto size = symbols.find(sizeSym);
    if (start == symbols.end() || size == symbols.end()) return std::nullopt;
    MetricsElfRegion region;
    region.start = start->second;
    region.size = size->second;
    return region;
  };
  info.itcm = lookupRegion("__itcm_start", "__itcm_size");
  info.dtcm = lookupRegion("__dtcm_start", "__dtcm_size");

  // Classify the allocated sections by region. The other sections (debug
  // info, symbol tables) are not loaded, so they take no memory.
  for (const llvm::object::SectionRef &section : elfObject->sections()) {
    llvm::object::ELFSectionRef elfSection(section);
    if (!(elfSection.getFlags() & llvm::ELF::SHF_ALLOC)) continue;
    if (section.getSize() == 0) continue;
    llvm::Expected<llvm::StringRef> name = section.getName();
    if (!name) return name.takeError();

    MetricsElfSection entry;
    entry.name = name->str();
    entry.address = section.getAddress();
    entry.size = section.getSize();
    entry.noBits = elfSection.getType() == llvm::ELF::SHT_NOBITS;
    if (regionContains(info.itcm, entry.address, entry.size)) {
      entry.region = "itcm";
    } else if (regionContains(info.dtcm, entry.address, entry.size)) {
      entry.region = "dtcm";
      if (entry.name == ".stack") info.dtcm->stack += entry.size;
    } else {
      entry.region = "other";
    }
    info.sections.push_back(std::move(entry));
  }

  // Count the used bytes of ITCM and DTCM from the PT_LOAD segments, matching
  // metrics/runtime/firmware_memory.c. The heap and stack sections are linked
  // into a DTCM segment but are reported separately, so they are excluded from
  // dtcm.used.
  auto phdrsOr = elfObject->getELFFile().program_headers();
  if (!phdrsOr) return phdrsOr.takeError();
  for (const auto &phdr : *phdrsOr) {
    if (phdr.p_type != llvm::ELF::PT_LOAD || phdr.p_memsz == 0) continue;
    const uint64_t paddr = phdr.p_paddr;
    const uint64_t memsz = phdr.p_memsz;
    if (regionContains(info.itcm, paddr, memsz)) {
      info.itcm->used += memsz;
      continue;
    }
    if (!regionContains(info.dtcm, paddr, memsz)) continue;
    uint64_t excluded = 0;
    for (const MetricsElfSection &section : info.sections) {
      if (section.region == "dtcm" &&
          (section.name == ".heap" || section.name == ".stack")) {
        excluded += rangeOverlap(paddr, memsz, section.address, section.size);
      }
    }
    info.dtcm->used += memsz >= excluded ? memsz - excluded : 0;
  }
  return info;
}

}  // namespace mlir::coralnpu_compiler
