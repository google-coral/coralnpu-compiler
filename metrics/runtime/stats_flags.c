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

#include "metrics/runtime/stats_flags.h"

#include <stdlib.h>
#include <string.h>

#include "iree/base/tooling/flags.h"
#include "metrics/runtime/stats_writer.h"

IREE_FLAG(string, coralnpu_stats_format, "none",
          "Per-dispatch execution statistics of CoralNPU devices: none, pretty "
          "(a table summarizing each exported function, written when the "
          "device is released) or json (one coralnpu.stats.v1 JSON object per "
          "dispatch, one per line). Overridden by CORALNPU_STATS_FORMAT unless "
          "set to another format.");

IREE_FLAG(string, coralnpu_stats_file, "",
          "Output file of --coralnpu_stats_format; empty writes to stderr and "
          "'-' writes to stdout. Output is appended to an existing file. "
          "Overridden by CORALNPU_STATS_FILE unless set.");

IREE_FLAG(string, coralnpu_stats_collect, "",
          "Comma-separated opt-in collectors of --coralnpu_stats_format: "
          "stack (the peak stack use of each dispatch; slow on Verilator). "
          "Overridden by CORALNPU_STATS_COLLECT unless set.");

// Returns |flag|, or the environment variable |env_name| if it is set and
// |flag| has its default value |default_value|; the same rule as the
// --simulator flag (runtime/driver/registration/driver_module.c).
static const char* iree_hal_coralnpu_stats_flag_or_env(
    const char* flag, const char* default_value, const char* env_name) {
  const char* value = getenv(env_name);
  return value && *value && strcmp(flag, default_value) == 0 ? value : flag;
}

// Options of the statistics writer, set from the flags. Static, as the writer
// borrows them for the lifetime of the drivers.
static iree_hal_coralnpu_stats_writer_options_t
    iree_hal_coralnpu_stats_writer_options;

iree_status_t iree_hal_coralnpu_stats_writer_from_flags(
    iree_hal_coralnpu_dispatch_observer_t* out_observer) {
  memset(out_observer, 0, sizeof(*out_observer));
  iree_hal_coralnpu_stats_writer_options_t* options =
      &iree_hal_coralnpu_stats_writer_options;
  IREE_RETURN_IF_ERROR(iree_hal_coralnpu_stats_format_parse(
      iree_make_cstring_view(iree_hal_coralnpu_stats_flag_or_env(
          FLAG_coralnpu_stats_format, "none", "CORALNPU_STATS_FORMAT")),
      &options->format));
  options->file_path =
      iree_make_cstring_view(iree_hal_coralnpu_stats_flag_or_env(
          FLAG_coralnpu_stats_file, "", "CORALNPU_STATS_FILE"));
  // Parsed even when the format is none, so unknown collectors are rejected.
  IREE_RETURN_IF_ERROR(iree_hal_coralnpu_stats_collect_parse(
      iree_make_cstring_view(iree_hal_coralnpu_stats_flag_or_env(
          FLAG_coralnpu_stats_collect, "", "CORALNPU_STATS_COLLECT")),
      &options->collect_fields));
  if (options->format == IREE_HAL_CORALNPU_STATS_FORMAT_NONE) {
    return iree_ok_status();
  }
  *out_observer = iree_hal_coralnpu_stats_writer_observer(options);
  return iree_ok_status();
}
