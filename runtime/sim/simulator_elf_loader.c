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

#include "runtime/sim/simulator_elf_loader.h"

#include <inttypes.h>

#include "iree/hal/local/elf/elf_types.h"
#include "runtime/driver/coralnpu_executable.h"
#include "runtime/sim/simulator_api.h"

static bool iree_hal_coralnpu_range_fits(uint32_t base, uint32_t size,
                                         uint32_t address, size_t length) {
  uint64_t begin = address;
  uint64_t end = begin + length;
  uint64_t region_begin = base;
  uint64_t region_end = region_begin + size;

  return begin >= region_begin && end >= begin && end <= region_end;
}

void iree_hal_coralnpu_simulator_zero_mem(coralnpu_simulator_t *sim,
                                          uint32_t address, size_t size) {
  static const uint8_t zeros[16384] = {0};
  while (size != 0) {
    size_t chunk = size < sizeof(zeros) ? size : sizeof(zeros);
    coralnpu_simulator_write_mem(sim, address, zeros, chunk);
    address += (uint32_t)chunk;
    size -= chunk;
  }
}

// FNV-1a hash over the whole ELF image.
static uint64_t iree_hal_coralnpu_hash_elf(iree_const_byte_span_t elf_image) {
  uint64_t h = 14695981039346656037ull;
  for (size_t i = 0; i < elf_image.data_length; ++i) {
    h = (h ^ elf_image.data[i]) * 1099511628211ull;
  }
  return h;
}

// Writes |phdr|'s data and zeroes the rest, stopping at |heap_start| if it is
// in the zeroed part.
static void iree_hal_coralnpu_load_segment(coralnpu_simulator_t *sim,
                                           const uint8_t *data,
                                           const iree_elf32_phdr_t *phdr,
                                           uint32_t heap_start) {
  if (phdr->p_filesz != 0) {
    coralnpu_simulator_write_mem(sim, phdr->p_paddr, data + phdr->p_offset,
                                 phdr->p_filesz);
  }
  uint32_t zero_begin = phdr->p_paddr + phdr->p_filesz;
  uint32_t zero_end = phdr->p_paddr + phdr->p_memsz;
  // The runtime writes the heap data it uses, and the stack (after the heap)
  // is the firmware's own.
  if (heap_start >= zero_begin && heap_start < zero_end) zero_end = heap_start;
  iree_hal_coralnpu_simulator_zero_mem(sim, zero_begin, zero_end - zero_begin);
}

// Reads an optional |symbol_name|, leaving |*out_value| untouched if absent.
static void iree_hal_coralnpu_read_optional_symbol(
    iree_const_byte_span_t elf_image, const char *symbol_name,
    uint32_t *out_value) {
  iree_status_ignore(iree_hal_coralnpu_executable_find_symbol(
      elf_image, symbol_name, out_value, NULL));
}

iree_status_t iree_hal_coralnpu_simulator_load_elf_with_layout(
    coralnpu_simulator_t *sim, iree_const_byte_span_t elf_image,
    iree_hal_coralnpu_simulator_elf_layout_t *out_layout) {
  IREE_ASSERT_ARGUMENT(out_layout);

  // Guarded by |iree_hal_coralnpu_simulator_mutex| in simulator_backend.c.
  static struct {
    const uint8_t *data;
    iree_host_size_t len;
    uint64_t hash;
    iree_hal_coralnpu_simulator_elf_layout_t layout;
  } elf_meta_cache[256];
  static uint32_t elf_meta_count = 0;

  const uint8_t *data = elf_image.data;
  const iree_elf32_ehdr_t *ehdr = (const iree_elf32_ehdr_t *)data;
  const uint64_t elf_hash = iree_hal_coralnpu_hash_elf(elf_image);
  for (uint32_t c = 0; c < elf_meta_count; ++c) {
    if (elf_meta_cache[c].data == data &&
        elf_meta_cache[c].len == elf_image.data_length &&
        elf_meta_cache[c].hash == elf_hash) {
      for (uint16_t i = 0; i < ehdr->e_phnum; ++i) {
        const iree_elf32_phdr_t *phdr =
            (const iree_elf32_phdr_t *)(data + ehdr->e_phoff +
                                        i * sizeof(iree_elf32_phdr_t));
        if (phdr->p_type == IREE_ELF_PT_LOAD && phdr->p_memsz != 0) {
          iree_hal_coralnpu_load_segment(
              sim, data, phdr, elf_meta_cache[c].layout.heap_start_addr);
        }
      }
      *out_layout = elf_meta_cache[c].layout;
      return iree_ok_status();
    }
  }

  // Defaults for images that do not export the TCM layout symbols.
  uint32_t itcm_start = 0x00000000u;
  uint32_t itcm_size = 8 * 1024;
  uint32_t dtcm_start = 0x00010000u;
  uint32_t dtcm_size = 32 * 1024;
  iree_hal_coralnpu_read_optional_symbol(elf_image, "__itcm_start",
                                         &itcm_start);
  iree_hal_coralnpu_read_optional_symbol(elf_image, "__itcm_size", &itcm_size);
  iree_hal_coralnpu_read_optional_symbol(elf_image, "__dtcm_start",
                                         &dtcm_start);
  iree_hal_coralnpu_read_optional_symbol(elf_image, "__dtcm_size", &dtcm_size);
  IREE_RETURN_IF_ERROR(iree_hal_coralnpu_executable_find_symbol(
      elf_image, "__heap_start", &out_layout->heap_start_addr, NULL));

  bool entry_is_executable = false;

  for (uint16_t i = 0; i < ehdr->e_phnum; ++i) {
    const iree_elf32_phdr_t *phdr =
        (const iree_elf32_phdr_t *)(data + ehdr->e_phoff +
                                    i * sizeof(iree_elf32_phdr_t));

    if (phdr->p_type != IREE_ELF_PT_LOAD || phdr->p_memsz == 0) {
      continue;
    }

    if (!iree_hal_coralnpu_range_fits(itcm_start, itcm_size, phdr->p_paddr,
                                      phdr->p_memsz) &&
        !iree_hal_coralnpu_range_fits(dtcm_start, dtcm_size, phdr->p_paddr,
                                      phdr->p_memsz)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "ELF PT_LOAD segment at 0x%08" PRIx32
                              " size=%" PRIu32 " is outside ITCM/DTCM",
                              phdr->p_paddr, phdr->p_memsz);
    }

    if ((phdr->p_flags & IREE_ELF_PF_X) != 0 &&
        ehdr->e_entry >= phdr->p_paddr &&
        (uint64_t)ehdr->e_entry < (uint64_t)phdr->p_paddr + phdr->p_memsz) {
      entry_is_executable = true;
    }

    iree_hal_coralnpu_load_segment(sim, data, phdr,
                                   out_layout->heap_start_addr);
  }

  if (!entry_is_executable) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "ELF entry point 0x%08" PRIx32
                            " is not inside an executable PT_LOAD segment",
                            ehdr->e_entry);
  }

  out_layout->start_pc = ehdr->e_entry;

  IREE_RETURN_IF_ERROR(iree_hal_coralnpu_executable_find_symbol(
      elf_image, "coralnpu_dispatch_request",
      &out_layout->dispatch_request_addr, &out_layout->dispatch_request_size));

  IREE_RETURN_IF_ERROR(iree_hal_coralnpu_executable_find_symbol(
      elf_image, "__heap_end", &out_layout->heap_end_addr, NULL));

  if (!iree_hal_coralnpu_range_fits(dtcm_start, dtcm_size,
                                    out_layout->dispatch_request_addr,
                                    out_layout->dispatch_request_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "CoralNPU dispatch request is outside DTCM");
  }

  if (out_layout->heap_end_addr < out_layout->heap_start_addr ||
      !iree_hal_coralnpu_range_fits(
          dtcm_start, dtcm_size, out_layout->heap_start_addr,
          out_layout->heap_end_addr - out_layout->heap_start_addr)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "ELF heap is outside DTCM");
  }

  if (elf_meta_count < IREE_ARRAYSIZE(elf_meta_cache)) {
    elf_meta_cache[elf_meta_count].data = data;
    elf_meta_cache[elf_meta_count].len = elf_image.data_length;
    elf_meta_cache[elf_meta_count].hash = elf_hash;
    elf_meta_cache[elf_meta_count].layout = *out_layout;
    ++elf_meta_count;
  }
  return iree_ok_status();
}
