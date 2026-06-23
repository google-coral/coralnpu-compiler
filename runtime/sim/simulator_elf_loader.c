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

#include <elf.h>
#include <inttypes.h>
#include <string.h>

#include "runtime/sim/simulator_api.h"

static bool iree_hal_coralnpu_simulator_is_elf32(
    iree_const_byte_span_t elf_image) {
  if (elf_image.data_length < sizeof(Elf32_Ehdr)) {
    return false;
  }

  const uint8_t *ident = (const uint8_t *)elf_image.data;

  return ident[EI_MAG0] == ELFMAG0 && ident[EI_MAG1] == ELFMAG1 &&
         ident[EI_MAG2] == ELFMAG2 && ident[EI_MAG3] == ELFMAG3 &&
         ident[EI_CLASS] == ELFCLASS32 && ident[EI_DATA] == ELFDATA2LSB;
}

static bool iree_hal_coralnpu_range_fits(uint32_t base, uint32_t size,
                                         uint32_t address, size_t length) {
  uint64_t begin = address;
  uint64_t end = begin + length;
  uint64_t region_begin = base;
  uint64_t region_end = region_begin + size;

  return begin >= region_begin && end >= begin && end <= region_end;
}

static iree_status_t iree_hal_coralnpu_validate_segment(uint32_t address,
                                                        size_t size) {
  if (size == 0) {
    return iree_ok_status();
  }

  if (iree_hal_coralnpu_range_fits(coralnpu_itcm_start, coralnpu_itcm_size,
                                   address, size) ||
      iree_hal_coralnpu_range_fits(coralnpu_dtcm_start, coralnpu_dtcm_size,
                                   address, size)) {
    return iree_ok_status();
  }

  return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                          "ELF PT_LOAD segment at 0x%08" PRIx32
                          " size=%zu is outside ITCM/DTCM",
                          address, size);
}

static iree_status_t iree_hal_coralnpu_copy_segment(uint32_t address,
                                                    const uint8_t *source,
                                                    size_t size) {
  if (size == 0) {
    return iree_ok_status();
  }

  if (iree_hal_coralnpu_range_fits(coralnpu_itcm_start, coralnpu_itcm_size,
                                   address, size)) {
    simulator_load_itcm(address - coralnpu_itcm_start, source, size);
    return iree_ok_status();
  }

  if (iree_hal_coralnpu_range_fits(coralnpu_dtcm_start, coralnpu_dtcm_size,
                                   address, size)) {
    simulator_load_dtcm(address - coralnpu_dtcm_start, source, size);
    return iree_ok_status();
  }

  return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                          "ELF segment at 0x%08" PRIx32
                          " size=%zu is outside ITCM/DTCM",
                          address, size);
}

static bool iree_hal_coralnpu_string_equals(const char *string_table,
                                            size_t string_table_size,
                                            uint32_t string_offset,
                                            const char *expected) {
  if (string_offset >= string_table_size) {
    return false;
  }

  const char *string = string_table + string_offset;
  size_t remaining = string_table_size - string_offset;

  return memchr(string, '\0', remaining) != NULL &&
         strcmp(string, expected) == 0;
}

static iree_status_t iree_hal_coralnpu_find_symbol(
    iree_const_byte_span_t elf_image, const char *symbol_name,
    uint32_t *out_address, uint32_t *out_size) {
  const uint8_t *data = (const uint8_t *)elf_image.data;
  const Elf32_Ehdr *ehdr = (const Elf32_Ehdr *)data;

  if (ehdr->e_shentsize != sizeof(Elf32_Shdr)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unexpected ELF section header size");
  }

  uint64_t section_table_end =
      (uint64_t)ehdr->e_shoff + (uint64_t)ehdr->e_shnum * sizeof(Elf32_Shdr);

  if (section_table_end > elf_image.data_length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "ELF section header table is out of bounds");
  }

  const Elf32_Shdr *section_headers =
      (const Elf32_Shdr *)(data + ehdr->e_shoff);

  for (uint16_t i = 0; i < ehdr->e_shnum; ++i) {
    const Elf32_Shdr *symbol_section = &section_headers[i];

    if (symbol_section->sh_type != SHT_SYMTAB) {
      continue;
    }

    if (symbol_section->sh_entsize != sizeof(Elf32_Sym) ||
        symbol_section->sh_link >= ehdr->e_shnum) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid ELF symbol table");
    }

    const Elf32_Shdr *string_section =
        &section_headers[symbol_section->sh_link];

    uint64_t symbol_table_end =
        (uint64_t)symbol_section->sh_offset + symbol_section->sh_size;

    uint64_t string_table_end =
        (uint64_t)string_section->sh_offset + string_section->sh_size;

    if (symbol_table_end > elf_image.data_length ||
        string_table_end > elf_image.data_length) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "ELF symbol table is out of bounds");
    }

    const Elf32_Sym *symbols =
        (const Elf32_Sym *)(data + symbol_section->sh_offset);

    size_t symbol_count = symbol_section->sh_size / sizeof(Elf32_Sym);

    const char *string_table = (const char *)(data + string_section->sh_offset);

    for (size_t j = 0; j < symbol_count; ++j) {
      const Elf32_Sym *symbol = &symbols[j];

      if (symbol->st_shndx == SHN_UNDEF) {
        continue;
      }

      if (!iree_hal_coralnpu_string_equals(string_table,
                                           string_section->sh_size,
                                           symbol->st_name, symbol_name)) {
        continue;
      }

      *out_address = symbol->st_value;

      if (out_size) {
        *out_size = symbol->st_size;
      }

      return iree_ok_status();
    }
  }

  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "required ELF symbol `%s` was not found",
                          symbol_name);
}

iree_status_t iree_hal_coralnpu_simulator_load_elf_with_layout(
    iree_const_byte_span_t elf_image,
    iree_hal_coralnpu_simulator_elf_layout_t *out_layout) {
  IREE_ASSERT_ARGUMENT(out_layout);
  memset(out_layout, 0, sizeof(*out_layout));

  if (!iree_hal_coralnpu_simulator_is_elf32(elf_image)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "dispatch image is not little-endian ELF32");
  }

  const uint8_t *data = (const uint8_t *)elf_image.data;
  const Elf32_Ehdr *ehdr = (const Elf32_Ehdr *)data;

  if (ehdr->e_machine != EM_RISCV) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "ELF is not RISC-V");
  }

  if (ehdr->e_phentsize != sizeof(Elf32_Phdr)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unexpected ELF program header size");
  }

  uint64_t program_table_end =
      (uint64_t)ehdr->e_phoff + (uint64_t)ehdr->e_phnum * sizeof(Elf32_Phdr);

  if (program_table_end > elf_image.data_length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "ELF program header table is out of bounds");
  }

  bool entry_is_executable = false;

  for (uint16_t i = 0; i < ehdr->e_phnum; ++i) {
    const Elf32_Phdr *phdr =
        (const Elf32_Phdr *)(data + ehdr->e_phoff + i * sizeof(Elf32_Phdr));

    if (phdr->p_type != PT_LOAD) {
      continue;
    }

    if (phdr->p_memsz < phdr->p_filesz) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "ELF PT_LOAD memsz is smaller than filesz");
    }

    uint64_t segment_file_end = (uint64_t)phdr->p_offset + phdr->p_filesz;

    if (segment_file_end > elf_image.data_length) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "ELF PT_LOAD data is out of bounds");
    }

    IREE_RETURN_IF_ERROR(
        iree_hal_coralnpu_validate_segment(phdr->p_paddr, phdr->p_memsz));

    if ((phdr->p_flags & PF_X) != 0 && ehdr->e_entry >= phdr->p_paddr &&
        (uint64_t)ehdr->e_entry < (uint64_t)phdr->p_paddr + phdr->p_memsz) {
      entry_is_executable = true;
    }

    IREE_RETURN_IF_ERROR(iree_hal_coralnpu_copy_segment(
        phdr->p_paddr, data + phdr->p_offset, phdr->p_filesz));

    if (phdr->p_memsz > phdr->p_filesz) {
      uint32_t zero_address = phdr->p_paddr + phdr->p_filesz;

      size_t remaining = phdr->p_memsz - phdr->p_filesz;

      uint8_t zero_buffer[256] = {0};

      while (remaining != 0) {
        size_t chunk =
            remaining < sizeof(zero_buffer) ? remaining : sizeof(zero_buffer);

        IREE_RETURN_IF_ERROR(
            iree_hal_coralnpu_copy_segment(zero_address, zero_buffer, chunk));

        zero_address += (uint32_t)chunk;
        remaining -= chunk;
      }
    }
  }

  if (!entry_is_executable) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "ELF entry point 0x%08" PRIx32
                            " is not inside an executable PT_LOAD segment",
                            ehdr->e_entry);
  }

  out_layout->start_pc = ehdr->e_entry;

  IREE_RETURN_IF_ERROR(iree_hal_coralnpu_find_symbol(
      elf_image, "coralnpu_dispatch_request",
      &out_layout->dispatch_request_addr, &out_layout->dispatch_request_size));

  IREE_RETURN_IF_ERROR(iree_hal_coralnpu_find_symbol(
      elf_image, "__heap_start", &out_layout->heap_start_addr, NULL));

  IREE_RETURN_IF_ERROR(iree_hal_coralnpu_find_symbol(
      elf_image, "__heap_end", &out_layout->heap_end_addr, NULL));

  if (!iree_hal_coralnpu_range_fits(coralnpu_dtcm_start, coralnpu_dtcm_size,
                                    out_layout->dispatch_request_addr,
                                    out_layout->dispatch_request_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "CoralNPU dispatch request is outside DTCM");
  }

  if (out_layout->heap_end_addr < out_layout->heap_start_addr ||
      !iree_hal_coralnpu_range_fits(
          coralnpu_dtcm_start, coralnpu_dtcm_size, out_layout->heap_start_addr,
          out_layout->heap_end_addr - out_layout->heap_start_addr)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "ELF heap is outside DTCM");
  }

  return iree_ok_status();
}

iree_status_t iree_hal_coralnpu_simulator_load_elf(
    iree_const_byte_span_t elf_image, uint32_t *out_start_pc) {
  IREE_ASSERT_ARGUMENT(out_start_pc);

  iree_hal_coralnpu_simulator_elf_layout_t layout;

  IREE_RETURN_IF_ERROR(
      iree_hal_coralnpu_simulator_load_elf_with_layout(elf_image, &layout));

  *out_start_pc = layout.start_pc;
  return iree_ok_status();
}
