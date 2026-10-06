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

#ifndef CRT_CORALNPU_DISPATCH_H_
#define CRT_CORALNPU_DISPATCH_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// The host runs every dispatch by loading its "firmware" on the core: the
// bare-metal ELF that the compiler links from the executable library,
// libcoralnpu_iree.a (coralnpu_main.c), and libcoralnpu_crt.a. The host and
// the firmware communicate through the firmware's
// `coralnpu_dispatch_request` variable: the host writes the whole request
// before every run, and reads it back after the firmware halted.

#define CORALNPU_DISPATCH_MAGIC 0x434E5055u
#define CORALNPU_DISPATCH_VERSION 1u

typedef enum coralnpu_dispatch_status_t {
  CORALNPU_DISPATCH_STATUS_EMPTY = 0,
  CORALNPU_DISPATCH_STATUS_READY = 1,
  CORALNPU_DISPATCH_STATUS_RUNNING = 2,
  CORALNPU_DISPATCH_STATUS_COMPLETE = 3,
  CORALNPU_DISPATCH_STATUS_ERROR = 4,
} coralnpu_dispatch_status_t;

// Bits of `coralnpu_dispatch_request_t::valid_mask`.
typedef enum coralnpu_dispatch_valid_t {
  CORALNPU_DISPATCH_VALID_KERNEL_CYCLES = 1u << 0,   // kernel_cycles_{lo,hi}
  CORALNPU_DISPATCH_VALID_KERNEL_INSTRET = 1u << 1,  // kernel_instret_{lo,hi}
} coralnpu_dispatch_valid_t;

typedef struct coralnpu_dispatch_request_t {
  uint32_t magic;
  uint16_t version;
  uint16_t status;

  int32_t return_code;

  uint32_t workgroup_size_x;
  uint32_t workgroup_size_y;
  uint32_t workgroup_size_z;

  uint32_t workgroup_count_x;
  uint32_t workgroup_count_y;
  uint32_t workgroup_count_z;

  uint16_t max_concurrency;
  uint8_t push_constant_count;
  uint8_t binding_count;

  uint32_t ordinal;
  uint32_t push_constants_addr;
  uint32_t binding_ptrs_addr;
  uint32_t binding_lengths_addr;
  // The workgroup local memory (in DTCM): its address and its size in bytes.
  // Both are 0 if the dispatch uses no local memory.
  uint32_t local_memory_addr;
  uint32_t local_memory_size;

  // Firmware outputs, which the host writes as zero.

  // Bitfield of CORALNPU_DISPATCH_VALID_* values: which of the fields below
  // the firmware filled.
  uint32_t valid_mask;

  // mcycle and minstret deltas around the workgroup loop, without the crt
  // startup and the request validation.
  uint32_t kernel_cycles_lo;
  uint32_t kernel_cycles_hi;
  uint32_t kernel_instret_lo;
  uint32_t kernel_instret_hi;
} coralnpu_dispatch_request_t;

#if defined(__cplusplus)
static_assert(sizeof(coralnpu_dispatch_request_t) == 84,
              "unexpected CoralNPU dispatch ABI size");
#else
_Static_assert(sizeof(coralnpu_dispatch_request_t) == 84,
               "unexpected CoralNPU dispatch ABI size");
#endif

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // CRT_CORALNPU_DISPATCH_H_
