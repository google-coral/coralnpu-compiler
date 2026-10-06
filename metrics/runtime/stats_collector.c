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

#include "metrics/runtime/stats_collector.h"

#include <string.h>

#include "crt/coralnpu_dispatch.h"
#include "iree/base/api.h"

void iree_hal_coralnpu_stats_collector_initialize(
    uint32_t collect_fields, const iree_hal_coralnpu_memory_stats_t* memory,
    iree_hal_coralnpu_stats_collector_t* out_collector) {
  memset(out_collector, 0, sizeof(*out_collector));
  if (memory) {
    out_collector->stats.memory = *memory;
    out_collector->has_memory = true;
  }
  out_collector->measure_stack = iree_all_bits_set(
      collect_fields, IREE_HAL_CORALNPU_DISPATCH_STATS_FIELD_STACK_PEAK);
}

// Records the memory footprint of the firmware and the dispatch data of |run|
// and, if requested, paints the stack to measure its peak use; the
// before_run hook of the iree_hal_coralnpu_stats_collector_t |self|. Reads
// the cycle counter last, so that only the run is counted.
static void iree_hal_coralnpu_stats_collector_before_run(
    void* self, const iree_hal_coralnpu_run_t* run) {
  iree_hal_coralnpu_stats_collector_t* collector =
      (iree_hal_coralnpu_stats_collector_t*)self;
  iree_hal_coralnpu_dispatch_stats_t* stats = &collector->stats;
  if (collector->has_memory) {
    iree_hal_coralnpu_memory_stats_t* memory = &stats->memory;
    const uint32_t heap_cursor = iree_max(run->heap_cursor, memory->heap.start);
    memory->heap.used = heap_cursor - memory->heap.start;
    stats->valid_fields |= IREE_HAL_CORALNPU_DISPATCH_STATS_FIELD_MEMORY;
    // Opt-in, as reading the stack back is slow on some backends.
    if (collector->measure_stack) {
      iree_hal_coralnpu_stack_paint(&run->core, memory, heap_cursor,
                                    &collector->stack_paint);
      collector->stack_painted = true;
    }
  }
  collector->cycle_start = run->core.cycle_count(run->core.self);
}

// Records the counters that the firmware reported in the dispatch request of
// |run|, which it reads back from the core.
static void iree_hal_coralnpu_stats_collector_read_request(
    iree_hal_coralnpu_stats_collector_t* collector,
    const iree_hal_coralnpu_run_t* run) {
  coralnpu_dispatch_request_t request;
  run->core.read_mem(run->core.self, run->dispatch_request_addr, &request,
                     sizeof(request));
  // A corrupted request reports nothing; the run fails on it.
  if (request.magic != CORALNPU_DISPATCH_MAGIC ||
      request.version != CORALNPU_DISPATCH_VERSION) {
    return;
  }
  iree_hal_coralnpu_dispatch_stats_t* stats = &collector->stats;
  if (iree_all_bits_set(request.valid_mask,
                        CORALNPU_DISPATCH_VALID_KERNEL_CYCLES)) {
    stats->cycles_kernel =
        ((uint64_t)request.kernel_cycles_hi << 32) | request.kernel_cycles_lo;
    stats->valid_fields |= IREE_HAL_CORALNPU_DISPATCH_STATS_FIELD_CYCLES_KERNEL;
  }
  if (iree_all_bits_set(request.valid_mask,
                        CORALNPU_DISPATCH_VALID_KERNEL_INSTRET)) {
    stats->instret_kernel =
        ((uint64_t)request.kernel_instret_hi << 32) | request.kernel_instret_lo;
    stats->valid_fields |=
        IREE_HAL_CORALNPU_DISPATCH_STATS_FIELD_INSTRET_KERNEL;
  }
}

// Records the cycles of the run of |run|, the counters of the firmware and the
// peak stack use; the after_run hook of the
// iree_hal_coralnpu_stats_collector_t |self|.
static void iree_hal_coralnpu_stats_collector_after_run(
    void* self, const iree_hal_coralnpu_run_t* run) {
  iree_hal_coralnpu_stats_collector_t* collector =
      (iree_hal_coralnpu_stats_collector_t*)self;
  iree_hal_coralnpu_dispatch_stats_t* stats = &collector->stats;
  // Read the cycle counter before the memory: reads may advance it.
  const uint64_t cycle_end = run->core.cycle_count(run->core.self);
  // Backends without a cycle counter always report 0.
  if (cycle_end > collector->cycle_start) {
    stats->cycles_total = cycle_end - collector->cycle_start;
    stats->valid_fields |= IREE_HAL_CORALNPU_DISPATCH_STATS_FIELD_CYCLES_TOTAL;
  }
  // The firmware reports its counters only when it completes, and the core
  // then halts.
  if (run->halted) {
    iree_hal_coralnpu_stats_collector_read_request(collector, run);
  }
  // Also when the dispatch failed, e.g. on a stack overflow.
  if (collector->stack_painted) {
    iree_hal_coralnpu_stack_scan(&run->core, &collector->stack_paint,
                                 &stats->memory.stack.used,
                                 &stats->memory.stack_overflow);
    stats->valid_fields |= IREE_HAL_CORALNPU_DISPATCH_STATS_FIELD_STACK_PEAK;
  }
}

iree_hal_coralnpu_run_hooks_t iree_hal_coralnpu_stats_collector_hooks(
    iree_hal_coralnpu_stats_collector_t* collector) {
  iree_hal_coralnpu_run_hooks_t hooks = {
      .self = collector,
      .before_run = iree_hal_coralnpu_stats_collector_before_run,
      .after_run = iree_hal_coralnpu_stats_collector_after_run,
  };
  return hooks;
}
