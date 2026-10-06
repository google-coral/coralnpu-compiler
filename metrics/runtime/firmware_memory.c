/*
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "metrics/runtime/firmware_memory.h"

#include <string.h>

#include "iree/hal/local/elf/elf_types.h"
#include "runtime/driver/coralnpu_executable.h"

// Sets |out_value| to the value of |symbol_name| in |image|. Returns false if
// |image| does not define the symbol.
static bool iree_hal_coralnpu_firmware_symbol(iree_const_byte_span_t image,
                                              const char* symbol_name,
                                              uint32_t* out_value) {
  iree_status_t status = iree_hal_coralnpu_executable_find_symbol(
      image, symbol_name, out_value, /*out_size=*/NULL);
  const bool found = iree_status_is_ok(status);
  iree_status_ignore(status);
  return found;
}

// Returns whether [address, address + length) is in |region|.
static bool iree_hal_coralnpu_region_contains(
    const iree_hal_coralnpu_memory_region_t* region, uint32_t address,
    uint32_t length) {
  return address >= region->start &&
         (uint64_t)address + length <= (uint64_t)region->start + region->size;
}

// Returns the number of bytes that [address, address + length) has in common
// with |region|.
static uint64_t iree_hal_coralnpu_region_overlap(
    const iree_hal_coralnpu_memory_region_t* region, uint32_t address,
    uint32_t length) {
  const uint64_t begin = iree_max((uint64_t)address, (uint64_t)region->start);
  const uint64_t end = iree_min((uint64_t)address + length,
                                (uint64_t)region->start + region->size);
  return end > begin ? end - begin : 0;
}

// Sets the |used| bytes of ITCM and DTCM in |memory| to the sizes of the
// PT_LOAD segments of |image| in them. The heap and the stack are linked into
// a DTCM segment but are reported separately, so they are left out of DTCM.
static void iree_hal_coralnpu_firmware_memory_count_segments(
    iree_const_byte_span_t image, iree_hal_coralnpu_memory_stats_t* memory) {
  const iree_elf32_ehdr_t* ehdr = (const iree_elf32_ehdr_t*)image.data;
  uint64_t itcm_used = 0;
  uint64_t dtcm_used = 0;
  for (uint16_t i = 0; i < ehdr->e_phnum; ++i) {
    const iree_elf32_phdr_t* phdr =
        (const iree_elf32_phdr_t*)(image.data + ehdr->e_phoff +
                                   i * sizeof(iree_elf32_phdr_t));
    if (phdr->p_type != IREE_ELF_PT_LOAD || phdr->p_memsz == 0) continue;
    // The heap and the stack are in DTCM, so an ITCM segment counts in full.
    if (iree_hal_coralnpu_region_contains(&memory->itcm, phdr->p_paddr,
                                          phdr->p_memsz)) {
      itcm_used += phdr->p_memsz;
      continue;
    }
    // The loader rejects segments outside the TCMs.
    if (!iree_hal_coralnpu_region_contains(&memory->dtcm, phdr->p_paddr,
                                           phdr->p_memsz)) {
      continue;
    }
    const uint64_t excluded = iree_hal_coralnpu_region_overlap(
                                  &memory->heap, phdr->p_paddr, phdr->p_memsz) +
                              iree_hal_coralnpu_region_overlap(
                                  &memory->stack, phdr->p_paddr, phdr->p_memsz);
    dtcm_used += phdr->p_memsz - excluded;
  }
  memory->itcm.used = (uint32_t)iree_min(itcm_used, UINT32_MAX);
  memory->dtcm.used = (uint32_t)iree_min(dtcm_used, UINT32_MAX);
}

bool iree_hal_coralnpu_firmware_memory_read(
    iree_const_byte_span_t image,
    iree_hal_coralnpu_memory_stats_t* out_memory) {
  memset(out_memory, 0, sizeof(*out_memory));
  if (image.data_length < sizeof(iree_elf32_ehdr_t)) return false;
  // The crt linker script defines all of them (crt/BUILD.bazel).
  uint32_t heap_end = 0;
  if (!iree_hal_coralnpu_firmware_symbol(image, "__itcm_start",
                                         &out_memory->itcm.start) ||
      !iree_hal_coralnpu_firmware_symbol(image, "__itcm_size",
                                         &out_memory->itcm.size) ||
      !iree_hal_coralnpu_firmware_symbol(image, "__dtcm_start",
                                         &out_memory->dtcm.start) ||
      !iree_hal_coralnpu_firmware_symbol(image, "__dtcm_size",
                                         &out_memory->dtcm.size) ||
      !iree_hal_coralnpu_firmware_symbol(image, "__heap_start",
                                         &out_memory->heap.start) ||
      !iree_hal_coralnpu_firmware_symbol(image, "__heap_end", &heap_end) ||
      !iree_hal_coralnpu_firmware_symbol(image, "__stack_start",
                                         &out_memory->stack.start) ||
      !iree_hal_coralnpu_firmware_symbol(image, "__stack_size",
                                         &out_memory->stack.size)) {
    return false;
  }
  out_memory->heap.size = heap_end - out_memory->heap.start;
  // The stack is painted to measure its peak, so it must be in DTCM after the
  // heap (which itself follows the static DTCM sections).
  if (heap_end < out_memory->heap.start || out_memory->stack.start < heap_end ||
      out_memory->stack.size == 0 ||
      !iree_hal_coralnpu_region_contains(
          &out_memory->dtcm, out_memory->heap.start, out_memory->heap.size) ||
      !iree_hal_coralnpu_region_contains(
          &out_memory->dtcm, out_memory->stack.start, out_memory->stack.size)) {
    return false;
  }
  iree_hal_coralnpu_firmware_memory_count_segments(image, out_memory);
  return true;
}
