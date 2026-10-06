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

#include <stddef.h>
#include <stdint.h>

#include "crt/coralnpu_dispatch.h"
#include "iree/hal/local/executable_library.h"

extern const iree_hal_executable_library_header_t*
iree_hal_executable_library_query(
    iree_hal_executable_library_version_t max_version,
    const iree_hal_executable_environment_v0_t* environment);

__attribute__((section(".noinit.coralnpu_dispatch_request"), aligned(16), used,
               visibility("default"))) coralnpu_dispatch_request_t
    coralnpu_dispatch_request;

// Non-const so the zero-initialized struct is placed in .bss (DTCM) rather
// than .rodata (ITCM), where the 8 KiB instruction budget is tight.
static iree_hal_executable_environment_v0_t environment;

__attribute__((noreturn)) static void coralnpu_halt(void) {
  __asm__ volatile(".word 0x08000073" ::: "memory");
  __builtin_unreachable();
}

// Records |return_code| and an error status in the dispatch request, and halts.
__attribute__((noinline, noreturn)) static void coralnpu_dispatch_fail(
    int32_t return_code) {
  coralnpu_dispatch_request.return_code = return_code;
  coralnpu_dispatch_request.status = CORALNPU_DISPATCH_STATUS_ERROR;
  coralnpu_halt();
}

// Reads the 64-bit mcycle counter. RV32 reads it in two halves; the read
// retries if the low half wrapped between the reads of the high half.
__attribute__((always_inline)) static inline uint64_t coralnpu_read_mcycle(
    void) {
  uint32_t hi, lo, hi_again;
  do {
    __asm__ volatile(
        "csrr %0, mcycleh\n"
        "csrr %1, mcycle\n"
        "csrr %2, mcycleh"
        : "=r"(hi), "=r"(lo), "=r"(hi_again)::"memory");
  } while (hi != hi_again);
  return ((uint64_t)hi << 32) | lo;
}

// Reads the 64-bit minstret counter, like coralnpu_read_mcycle.
__attribute__((always_inline)) static inline uint64_t coralnpu_read_minstret(
    void) {
  uint32_t hi, lo, hi_again;
  do {
    __asm__ volatile(
        "csrr %0, minstreth\n"
        "csrr %1, minstret\n"
        "csrr %2, minstreth"
        : "=r"(hi), "=r"(lo), "=r"(hi_again)::"memory");
  } while (hi != hi_again);
  return ((uint64_t)hi << 32) | lo;
}

// Runs all workgroups of the dispatch that the runtime wrote to
// coralnpu_dispatch_request, writes the status to the request, and halts.
void main(void) {
  coralnpu_dispatch_request_t* request = &coralnpu_dispatch_request;

  if (request->magic != CORALNPU_DISPATCH_MAGIC ||
      request->version != CORALNPU_DISPATCH_VERSION ||
      request->status != CORALNPU_DISPATCH_STATUS_READY) {
    coralnpu_dispatch_fail(/*return_code=*/-1);
  }

  // Each table or buffer that is not empty must have an address.
  if ((request->push_constant_count != 0 &&
       request->push_constants_addr == 0) ||
      (request->binding_count != 0 && (request->binding_ptrs_addr == 0 ||
                                       request->binding_lengths_addr == 0)) ||
      (request->local_memory_size != 0 && request->local_memory_addr == 0)) {
    coralnpu_dispatch_fail(/*return_code=*/-2);
  }

  request->status = CORALNPU_DISPATCH_STATUS_RUNNING;
  request->return_code = 0;

  const iree_hal_executable_library_header_t* library_header =
      iree_hal_executable_library_query(
          IREE_HAL_EXECUTABLE_LIBRARY_VERSION_LATEST, &environment);

  if (library_header == NULL) {
    coralnpu_dispatch_fail(/*return_code=*/-3);
  }

  const iree_hal_executable_library_v0_t* library =
      (const iree_hal_executable_library_v0_t*)library_header;

  if (library->exports.ptrs == NULL ||
      request->ordinal >= library->exports.count ||
      library->exports.ptrs[request->ordinal] == NULL) {
    coralnpu_dispatch_fail(/*return_code=*/-4);
  }

  const uint32_t workgroup_count_x = request->workgroup_count_x;
  const uint32_t workgroup_count_y = request->workgroup_count_y;
  const uint32_t workgroup_count_z = request->workgroup_count_z;

  iree_hal_executable_dispatch_state_v0_t dispatch_state = {
      .workgroup_size_x = request->workgroup_size_x,
      .workgroup_size_y = request->workgroup_size_y,
      .workgroup_size_z = request->workgroup_size_z,
      .constant_count = request->push_constant_count,
      .workgroup_count_x = workgroup_count_x,
      .workgroup_count_y = workgroup_count_y,
      .workgroup_count_z = workgroup_count_z,
      .max_concurrency = request->max_concurrency,
      .binding_count = request->binding_count,
      .constants = (const uint32_t*)(uintptr_t)request->push_constants_addr,
      .binding_ptrs = (void* const*)(uintptr_t)request->binding_ptrs_addr,
      .binding_lengths =
          (const size_t*)(uintptr_t)request->binding_lengths_addr,
  };

  iree_hal_executable_dispatch_v0_t dispatch =
      library->exports.ptrs[request->ordinal];

  // Partial initializers emit memset, which the freestanding build lacks.
  iree_hal_executable_workgroup_state_v0_t workgroup_state;
  workgroup_state.reserved = 0;
  workgroup_state.processor_id = 0;
  workgroup_state.local_memory = (void*)(uintptr_t)request->local_memory_addr;
  workgroup_state.local_memory_size = request->local_memory_size;

  // The kernel counters cover the workgroup loop only, without the crt
  // startup, the request validation and the library query.
  const uint64_t cycles_start = coralnpu_read_mcycle();
  const uint64_t instret_start = coralnpu_read_minstret();

  for (uint32_t z = 0; z < workgroup_count_z; ++z) {
    workgroup_state.workgroup_id_z = z;
    for (uint32_t y = 0; y < workgroup_count_y; ++y) {
      workgroup_state.workgroup_id_y = y;
      for (uint32_t x = 0; x < workgroup_count_x; ++x) {
        workgroup_state.workgroup_id_x = x;

        int result = dispatch(&environment, &dispatch_state, &workgroup_state);

        if (result != 0) {
          coralnpu_dispatch_fail(result);
        }
      }
    }
  }

  const uint64_t instret = coralnpu_read_minstret() - instret_start;
  const uint64_t cycles = coralnpu_read_mcycle() - cycles_start;
  request->kernel_cycles_lo = (uint32_t)cycles;
  request->kernel_cycles_hi = (uint32_t)(cycles >> 32);
  request->kernel_instret_lo = (uint32_t)instret;
  request->kernel_instret_hi = (uint32_t)(instret >> 32);
  request->valid_mask = CORALNPU_DISPATCH_VALID_KERNEL_CYCLES |
                        CORALNPU_DISPATCH_VALID_KERNEL_INSTRET;

  request->status = CORALNPU_DISPATCH_STATUS_COMPLETE;
  coralnpu_halt();
}
