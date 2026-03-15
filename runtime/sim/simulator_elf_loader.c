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
#include <string.h>

#include "runtime/sim/simulator_api.h"

static bool
iree_hal_coralnpu_simulator_is_elf32(iree_const_byte_span_t elf_image) {
  if (elf_image.data_length < sizeof(Elf32_Ehdr))
    return false;
  const uint8_t *ident = (const uint8_t *)elf_image.data;
  return ident[0] == 0x7f && ident[1] == 'E' && ident[2] == 'L' &&
         ident[3] == 'F' && ident[4] == ELFCLASS32;
}

static iree_status_t
iree_hal_coralnpu_simulator_copy_segment(uint32_t paddr, const uint8_t *src,
                                         size_t filesz) {
  if (filesz == 0)
    return iree_ok_status();

  // ITCM region
  if (paddr >= coralnpu_itcm_start &&
      paddr + filesz <= coralnpu_itcm_start + coralnpu_itcm_size) {
    simulator_load_itcm(paddr - coralnpu_itcm_start, src, filesz);
    return iree_ok_status();
  }

  // DTCM region
  if (paddr >= coralnpu_dtcm_start &&
      paddr + filesz <= coralnpu_dtcm_start + coralnpu_dtcm_size) {
    simulator_load_dtcm(paddr - coralnpu_dtcm_start, src, filesz);
    return iree_ok_status();
  }

  return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                          "ELF PT_LOAD segment at 0x%08" PRIx32
                          " size=%zu is outside ITCM/DTCM",
                          paddr, filesz);
}

iree_status_t
iree_hal_coralnpu_simulator_load_elf(iree_const_byte_span_t elf_image,
                                     uint32_t *out_start_pc) {
  IREE_ASSERT_ARGUMENT(out_start_pc);
  *out_start_pc = 0;

  if (!iree_hal_coralnpu_simulator_is_elf32(elf_image)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "dispatch image is not ELF32");
  }

  const uint8_t *data = (const uint8_t *)elf_image.data;
  const Elf32_Ehdr *ehdr = (const Elf32_Ehdr *)data;

  if (ehdr->e_machine != EM_RISCV) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "ELF is not RISC-V, e_machine=%u",
                            (unsigned)ehdr->e_machine);
  }

  if (ehdr->e_phentsize != sizeof(Elf32_Phdr)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unexpected program header size: %u",
                            (unsigned)ehdr->e_phentsize);
  }

  size_t ph_table_end =
      (size_t)ehdr->e_phoff + (size_t)ehdr->e_phnum * sizeof(Elf32_Phdr);
  if (ph_table_end > elf_image.data_length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "ELF program header table out of bounds");
  }

  uint32_t fallback_entry_pc = 0;

  for (uint16_t i = 0; i < ehdr->e_phnum; ++i) {
    const Elf32_Phdr *phdr =
        (const Elf32_Phdr *)(data + ehdr->e_phoff + i * sizeof(Elf32_Phdr));

    fprintf(stderr,
            "[SIM ELF] phdr[%u]: type=%u off=0x%08x paddr=0x%08x "
            "filesz=%u memsz=%u flags=0x%x\n",
            (unsigned)i, (unsigned)phdr->p_type, (unsigned)phdr->p_offset,
            (unsigned)phdr->p_paddr, (unsigned)phdr->p_filesz,
            (unsigned)phdr->p_memsz, (unsigned)phdr->p_flags);
    fflush(stderr);

    if (phdr->p_type == PT_LOAD && (phdr->p_flags & PF_X) &&
        fallback_entry_pc == 0) {
      fallback_entry_pc = phdr->p_paddr;
    }

    if (phdr->p_type != PT_LOAD)
      continue;
    if (phdr->p_filesz == 0)
      continue;

    size_t segment_end = (size_t)phdr->p_offset + (size_t)phdr->p_filesz;
    if (segment_end > elf_image.data_length) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "ELF PT_LOAD segment out of bounds");
    }

    IREE_RETURN_IF_ERROR(iree_hal_coralnpu_simulator_copy_segment(
        phdr->p_paddr, data + phdr->p_offset, phdr->p_filesz));

    if (phdr->p_memsz > phdr->p_filesz) {
      uint32_t zero_addr = phdr->p_paddr + phdr->p_filesz;
      size_t zero_size = phdr->p_memsz - phdr->p_filesz;

      uint8_t zero_buffer[256];
      memset(zero_buffer, 0, sizeof(zero_buffer));

      while (zero_size > 0) {
        size_t chunk =
            zero_size < sizeof(zero_buffer) ? zero_size : sizeof(zero_buffer);
        IREE_RETURN_IF_ERROR(iree_hal_coralnpu_simulator_copy_segment(
            zero_addr, zero_buffer, chunk));
        zero_addr += (uint32_t)chunk;
        zero_size -= chunk;
      }
    }
  }

  if (ehdr->e_entry != 0) {
    *out_start_pc = ehdr->e_entry;
  } else if (fallback_entry_pc != 0) {
    *out_start_pc = fallback_entry_pc;
  } else {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "ELF has no usable entry point");
  }
  return iree_ok_status();
}
