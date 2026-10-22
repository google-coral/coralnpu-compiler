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

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Bits of iree_hal_coralnpu_dispatch_stats_t::valid_fields.
enum iree_hal_coralnpu_dispatch_stats_field_bits_e {
  IREE_HAL_CORALNPU_DISPATCH_STATS_FIELD_NONE = 0u,
  // |cycles_total| is valid.
  IREE_HAL_CORALNPU_DISPATCH_STATS_FIELD_CYCLES_TOTAL = 1u << 0,
};

// Statistics of a single dispatch. Fields whose bit is not set in
// |valid_fields| were not collected and must be ignored.
typedef struct iree_hal_coralnpu_dispatch_stats_t {
  // Bitfield of IREE_HAL_CORALNPU_DISPATCH_STATS_FIELD_* values.
  uint32_t valid_fields;
  // Cycles from the start of the dispatch program until it terminated,
  // including the firmware startup and teardown code.
  uint64_t cycles_total;
} iree_hal_coralnpu_dispatch_stats_t;

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // METRICS_RUNTIME_DISPATCH_STATS_H_
