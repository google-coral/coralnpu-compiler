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

#ifndef RUNTIME_SIM_SIMULATOR_ELF_LOADER_H_
#define RUNTIME_SIM_SIMULATOR_ELF_LOADER_H_

#include <stdint.h>

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_coralnpu_simulator_elf_layout_t {
  uint32_t start_pc;

  uint32_t dispatch_request_addr;
  uint32_t dispatch_request_size;

  uint32_t heap_start_addr;
  uint32_t heap_end_addr;
} iree_hal_coralnpu_simulator_elf_layout_t;

iree_status_t iree_hal_coralnpu_simulator_load_elf_with_layout(
    iree_const_byte_span_t elf_image,
    iree_hal_coralnpu_simulator_elf_layout_t *out_layout);

// Backward-compatible wrapper.
iree_status_t iree_hal_coralnpu_simulator_load_elf(
    iree_const_byte_span_t elf_image, uint32_t *out_start_pc);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // RUNTIME_SIM_SIMULATOR_ELF_LOADER_H_
