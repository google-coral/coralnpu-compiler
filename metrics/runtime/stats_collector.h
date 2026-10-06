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

// Collects the statistics of a dispatch (dispatch_stats.h) through the run
// hooks of the execution backend that runs it
// (runtime/driver/coralnpu_run_hooks.h).

#ifndef METRICS_RUNTIME_STATS_COLLECTOR_H_
#define METRICS_RUNTIME_STATS_COLLECTOR_H_

#include <stdbool.h>
#include <stdint.h>

#include "metrics/runtime/dispatch_stats.h"
#include "metrics/runtime/stack_peak.h"
#include "runtime/driver/coralnpu_run_hooks.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_coralnpu_stats_collector_t {
  // The statistics collected so far.
  iree_hal_coralnpu_dispatch_stats_t stats;
  // The cycle counter of the core when the run started.
  uint64_t cycle_start;
  // Whether |stats.memory| has the layout of the firmware ELF.
  bool has_memory;
  // Whether the peak stack use is requested, whether the stack was painted to
  // measure it, and the painted memory.
  bool measure_stack;
  bool stack_painted;
  iree_hal_coralnpu_stack_paint_t stack_paint;
} iree_hal_coralnpu_stats_collector_t;

// Initializes |out_collector| with no statistics. |collect_fields| has the
// opt-in IREE_HAL_CORALNPU_DISPATCH_STATS_FIELD_* bits to collect besides the
// fields that the collector always collects; only
// IREE_HAL_CORALNPU_DISPATCH_STATS_FIELD_STACK_PEAK is opt-in. |memory| is the
// memory layout of the firmware ELF, or NULL if the ELF has none.
void iree_hal_coralnpu_stats_collector_initialize(
    uint32_t collect_fields, const iree_hal_coralnpu_memory_stats_t *memory,
    iree_hal_coralnpu_stats_collector_t *out_collector);

// Returns the run hooks that collect the statistics of a dispatch into
// |collector|, which must outlive the dispatch.
iree_hal_coralnpu_run_hooks_t iree_hal_coralnpu_stats_collector_hooks(
    iree_hal_coralnpu_stats_collector_t *collector);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // METRICS_RUNTIME_STATS_COLLECTOR_H_
