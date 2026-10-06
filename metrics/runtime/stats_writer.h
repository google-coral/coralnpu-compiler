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

#ifndef METRICS_RUNTIME_STATS_WRITER_H_
#define METRICS_RUNTIME_STATS_WRITER_H_

#include "iree/base/api.h"
#include "runtime/driver/coralnpu_dispatch_observer.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Schema identifier written in every JSON stats record. A record is one line
// with the keys below, which hold the fields of
// iree_hal_coralnpu_dispatch_stats_t (see dispatch_stats.h); a key is left out
// when its field is not valid. Adding keys keeps the schema version; renaming
// or removing keys bumps it.
//   schema         IREE_HAL_CORALNPU_STATS_SCHEMA
//   dispatch       {executable, name, ordinal}: the export that ran
//   backend        {simulator}: mpact, spike, verilator or fpga
//   status         {code}: the IREE status code, "OK" on success
//   cycles         {total, kernel}: cycles_total and cycles_kernel
//   instret        {kernel}: instret_kernel
//   memory         {itcm, dtcm, heap, stack}: memory; each region is
//                  {start, size, used}, except for the stack, which is
//                  {start, size, peak, overflow}: peak and overflow need the
//                  stack collector (iree_hal_coralnpu_stats_collect_parse).
#define IREE_HAL_CORALNPU_STATS_SCHEMA "coralnpu.stats.v1"

// Output format of per-dispatch execution statistics.
typedef enum iree_hal_coralnpu_stats_format_e {
  // No statistics are collected.
  IREE_HAL_CORALNPU_STATS_FORMAT_NONE = 0,
  // A human-readable table summarizing each exported function, written when
  // the device is destroyed.
  IREE_HAL_CORALNPU_STATS_FORMAT_PRETTY,
  // One JSON object per dispatch (schema IREE_HAL_CORALNPU_STATS_SCHEMA), one
  // object per line, written as each dispatch completes.
  IREE_HAL_CORALNPU_STATS_FORMAT_JSON,
} iree_hal_coralnpu_stats_format_t;

// Parses |value| ("none", "pretty" or "json") into |out_format|.
iree_status_t iree_hal_coralnpu_stats_format_parse(
    iree_string_view_t value, iree_hal_coralnpu_stats_format_t *out_format);

// Parses |value|, a comma-separated list of opt-in (costly) collectors, into
// the IREE_HAL_CORALNPU_DISPATCH_STATS_FIELD_* bits they enable. Collectors:
//   stack: peak stack use (IREE_HAL_CORALNPU_DISPATCH_STATS_FIELD_STACK_PEAK).
// An empty list enables none; unknown collectors are rejected.
iree_status_t iree_hal_coralnpu_stats_collect_parse(iree_string_view_t value,
                                                    uint32_t *out_fields);

// Options of a statistics writer.
typedef struct iree_hal_coralnpu_stats_writer_options_t {
  // Output format; must not be IREE_HAL_CORALNPU_STATS_FORMAT_NONE.
  iree_hal_coralnpu_stats_format_t format;
  // Output file path. Empty writes to stderr and "-" writes to stdout. Output
  // is appended to an existing file.
  iree_string_view_t file_path;
  // Opt-in IREE_HAL_CORALNPU_DISPATCH_STATS_FIELD_* bits to collect (see
  // iree_hal_coralnpu_stats_collect_parse).
  uint32_t collect_fields;
} iree_hal_coralnpu_stats_writer_options_t;

// Returns a dispatch observer that writes the statistics of the dispatches as
// described by |options|, which must outlive the devices using the observer.
// Each device opens the output when it is created and, with the pretty format,
// writes its summary when it is destroyed. Thread-safe.
iree_hal_coralnpu_dispatch_observer_t iree_hal_coralnpu_stats_writer_observer(
    const iree_hal_coralnpu_stats_writer_options_t *options);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // METRICS_RUNTIME_STATS_WRITER_H_
