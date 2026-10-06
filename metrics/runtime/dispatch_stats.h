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

// The statistics of a single dispatch: collected through the run hooks of the
// execution backend that runs it (stats_collector.h) and written by the
// statistics writer (stats_writer.h).

#ifndef METRICS_RUNTIME_DISPATCH_STATS_H_
#define METRICS_RUNTIME_DISPATCH_STATS_H_

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Bits of iree_hal_coralnpu_dispatch_stats_t::valid_fields.
enum iree_hal_coralnpu_dispatch_stats_field_bits_e {
  IREE_HAL_CORALNPU_DISPATCH_STATS_FIELD_NONE = 0u,
  // |cycles_total| is valid.
  IREE_HAL_CORALNPU_DISPATCH_STATS_FIELD_CYCLES_TOTAL = 1u << 0,
  // |cycles_kernel| is valid.
  IREE_HAL_CORALNPU_DISPATCH_STATS_FIELD_CYCLES_KERNEL = 1u << 1,
  // |instret_kernel| is valid.
  IREE_HAL_CORALNPU_DISPATCH_STATS_FIELD_INSTRET_KERNEL = 1u << 2,
  // |memory| is valid, except for |memory.stack.used| and
  // |memory.stack_overflow|.
  IREE_HAL_CORALNPU_DISPATCH_STATS_FIELD_MEMORY = 1u << 3,
  // |memory.stack.used| and |memory.stack_overflow| are valid. Opt-in: the
  // collector measures them only on request.
  IREE_HAL_CORALNPU_DISPATCH_STATS_FIELD_STACK_PEAK = 1u << 4,
};

// A memory region of the core and the number of bytes of it in use.
typedef struct iree_hal_coralnpu_memory_region_t {
  uint32_t start;
  uint32_t size;
  uint32_t used;
} iree_hal_coralnpu_memory_region_t;

// The memory footprint of a dispatch in the TCMs of the core: where the
// firmware of the dispatch and its data are, and how much of each region they
// use.
typedef struct iree_hal_coralnpu_memory_stats_t {
  // ITCM; |used| is the size of the PT_LOAD segments of the firmware in it.
  iree_hal_coralnpu_memory_region_t itcm;
  // DTCM; |used| is the size of the PT_LOAD segments of the firmware in it,
  // without the heap and the stack, which are reported separately.
  iree_hal_coralnpu_memory_region_t dtcm;
  // Heap of the firmware in DTCM, which holds the dispatch data; |used| is
  // the size of the dispatch data, with its alignment padding.
  iree_hal_coralnpu_memory_region_t heap;
  // Stack of the firmware in DTCM; |used| is the peak stack use. It exceeds
  // |size| when the stack overflowed into the guard area below it.
  iree_hal_coralnpu_memory_region_t stack;
  // The stack overflowed: a word below the stack was overwritten or, without
  // a guard area, the whole stack was used.
  bool stack_overflow;
} iree_hal_coralnpu_memory_stats_t;

// Statistics of a single dispatch. Fields whose bit is not set in
// |valid_fields| were not collected and must be ignored.
typedef struct iree_hal_coralnpu_dispatch_stats_t {
  // Bitfield of IREE_HAL_CORALNPU_DISPATCH_STATS_FIELD_* values.
  uint32_t valid_fields;
  // Cycles from the start of the dispatch program until it terminated,
  // including the firmware startup and teardown code.
  uint64_t cycles_total;
  // Cycles (mcycle) and retired instructions (minstret) of the workgroup
  // loop, measured by the firmware.
  uint64_t cycles_kernel;
  uint64_t instret_kernel;
  // Memory footprint of the firmware and the dispatch data.
  iree_hal_coralnpu_memory_stats_t memory;
} iree_hal_coralnpu_dispatch_stats_t;

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // METRICS_RUNTIME_DISPATCH_STATS_H_
