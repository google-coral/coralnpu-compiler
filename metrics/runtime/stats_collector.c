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

void iree_hal_coralnpu_stats_collector_initialize(
    iree_hal_coralnpu_stats_collector_t *out_collector) {
  memset(out_collector, 0, sizeof(*out_collector));
}

// Reads the cycle counter at the start of the run of |run|; the before_run
// hook of the iree_hal_coralnpu_stats_collector_t |self|.
static void iree_hal_coralnpu_stats_collector_before_run(
    void *self, const iree_hal_coralnpu_run_t *run) {
  iree_hal_coralnpu_stats_collector_t *collector =
      (iree_hal_coralnpu_stats_collector_t *)self;
  collector->cycle_start = run->core.cycle_count(run->core.self);
}

// Records the cycles of the run of |run|; the after_run hook of the
// iree_hal_coralnpu_stats_collector_t |self|.
static void iree_hal_coralnpu_stats_collector_after_run(
    void *self, const iree_hal_coralnpu_run_t *run) {
  iree_hal_coralnpu_stats_collector_t *collector =
      (iree_hal_coralnpu_stats_collector_t *)self;
  iree_hal_coralnpu_dispatch_stats_t *stats = &collector->stats;
  const uint64_t cycle_end = run->core.cycle_count(run->core.self);
  // Backends without a cycle counter always report 0.
  if (cycle_end > collector->cycle_start) {
    stats->cycles_total = cycle_end - collector->cycle_start;
    stats->valid_fields |= IREE_HAL_CORALNPU_DISPATCH_STATS_FIELD_CYCLES_TOTAL;
  }
}

iree_hal_coralnpu_run_hooks_t iree_hal_coralnpu_stats_collector_hooks(
    iree_hal_coralnpu_stats_collector_t *collector) {
  iree_hal_coralnpu_run_hooks_t hooks = {
      .self = collector,
      .before_run = iree_hal_coralnpu_stats_collector_before_run,
      .after_run = iree_hal_coralnpu_stats_collector_after_run,
  };
  return hooks;
}
