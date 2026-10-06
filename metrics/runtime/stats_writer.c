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

#include "metrics/runtime/stats_writer.h"

#include <errno.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include "iree/base/threading/mutex.h"
#include "metrics/runtime/dispatch_stats.h"
#include "metrics/runtime/firmware_memory.h"
#include "metrics/runtime/stats_collector.h"

// Writes the statistics of the dispatches of one device: the per-device
// context of the observer. Thread-safe.
typedef struct iree_hal_coralnpu_stats_t iree_hal_coralnpu_stats_t;

// Cached memory layout of one firmware ELF image.
typedef struct iree_hal_coralnpu_stats_image_t {
  iree_const_byte_span_t image;
  bool has_memory;
  iree_hal_coralnpu_memory_stats_t memory;
} iree_hal_coralnpu_stats_image_t;

// Aggregated statistics of one exported function, used by the pretty format.
typedef struct iree_hal_coralnpu_stats_entry_t {
  // Stored after the entry in the same allocation.
  iree_string_view_t executable_name;
  // Stored after the entry in the same allocation.
  iree_string_view_t export_name;
  uint32_t export_ordinal;
  uint64_t dispatch_count;
  uint64_t failure_count;
  // Number of dispatches that reported a cycle count.
  uint64_t cycles_count;
  uint64_t cycles_sum;
  uint64_t cycles_min;
  uint64_t cycles_max;
  // Number of dispatches that reported kernel cycles / instret, and sums.
  uint64_t kernel_cycles_count;
  uint64_t kernel_cycles_sum;
  uint64_t kernel_instret_count;
  uint64_t kernel_instret_sum;
  // Number of dispatches that reported their memory footprint, and the
  // largest ITCM, static DTCM data, and heap use.
  uint64_t memory_count;
  uint32_t itcm_used_max;
  uint32_t dtcm_used_max;
  uint32_t heap_used_max;
  // Number of dispatches that reported their stack peak, the largest one,
  // and whether any of them overflowed the stack.
  uint64_t stack_peak_count;
  uint32_t stack_peak_max;
  bool stack_overflow;
} iree_hal_coralnpu_stats_entry_t;

struct iree_hal_coralnpu_stats_t {
  iree_allocator_t host_allocator;
  iree_hal_coralnpu_stats_format_t format;
  // Opt-in IREE_HAL_CORALNPU_DISPATCH_STATS_FIELD_* bits to collect.
  uint32_t collect_fields;
  // Name of the execution backend; its storage trails the struct.
  iree_string_view_t backend_name;
  FILE *file;
  // True if |file| was opened by the writer and must be closed.
  bool owns_file;
  iree_slim_mutex_t mutex;
  // Guarded by |mutex|.
  iree_host_size_t image_count;
  iree_host_size_t image_capacity;
  iree_hal_coralnpu_stats_image_t *images;
  iree_host_size_t entry_count;
  iree_host_size_t entry_capacity;
  iree_hal_coralnpu_stats_entry_t **entries;
};

iree_status_t iree_hal_coralnpu_stats_format_parse(
    iree_string_view_t value, iree_hal_coralnpu_stats_format_t *out_format) {
  if (iree_string_view_is_empty(value) ||
      iree_string_view_equal(value, IREE_SV("none"))) {
    *out_format = IREE_HAL_CORALNPU_STATS_FORMAT_NONE;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("pretty"))) {
    *out_format = IREE_HAL_CORALNPU_STATS_FORMAT_PRETTY;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("json"))) {
    *out_format = IREE_HAL_CORALNPU_STATS_FORMAT_JSON;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unknown CoralNPU stats format '%.*s' (expected "
                          "'none', 'pretty' or 'json')",
                          (int)value.size, value.data);
}

iree_status_t iree_hal_coralnpu_stats_collect_parse(iree_string_view_t value,
                                                    uint32_t *out_fields) {
  *out_fields = 0;
  // Parse the comma-separated names one at a time.
  while (!iree_string_view_is_empty(value)) {
    iree_string_view_t name = iree_string_view_empty();
    iree_string_view_split(value, ',', &name, &value);
    name = iree_string_view_trim(name);
    // Ignore empty names, e.g. from a trailing comma.
    if (iree_string_view_is_empty(name)) continue;
    if (iree_string_view_equal(name, IREE_SV("stack"))) {
      *out_fields |= IREE_HAL_CORALNPU_DISPATCH_STATS_FIELD_STACK_PEAK;
      continue;
    }
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown CoralNPU stats collector '%.*s' "
                            "(expected 'stack')",
                            (int)name.size, name.data);
  }
  return iree_ok_status();
}

// Sets |out_file| to the stats output: stderr for an empty |file_path|,
// stdout for "-", else |file_path| opened for appending. Sets |out_owns_file|
// to true if the caller must close |out_file|.
static iree_status_t iree_hal_coralnpu_stats_open_file(
    iree_string_view_t file_path, iree_allocator_t host_allocator,
    FILE **out_file, bool *out_owns_file) {
  *out_owns_file = false;
  if (iree_string_view_is_empty(file_path)) {
    *out_file = stderr;
    return iree_ok_status();
  }
  if (iree_string_view_equal(file_path, IREE_SV("-"))) {
    *out_file = stdout;
    return iree_ok_status();
  }

  // fopen requires a NUL-terminated path.
  char *path = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, file_path.size + 1,
                                             (void **)&path));
  iree_string_view_to_cstring(file_path, path, file_path.size + 1);
  // Append: a process may create several devices (e.g. iree-check-module
  // creates one per test function) and each must not clobber the others.
  FILE *file = fopen(path, "a");
  const int open_errno = errno;
  iree_allocator_free(host_allocator, path);
  if (!file) {
    return iree_make_status(iree_status_code_from_errno(open_errno),
                            "failed to open CoralNPU stats file '%.*s': %s",
                            (int)file_path.size, file_path.data,
                            strerror(open_errno));
  }
  *out_file = file;
  *out_owns_file = true;
  return iree_ok_status();
}

// Creates the context of a device: opens the output described by the
// iree_hal_coralnpu_stats_writer_options_t |self|. |backend_name| is copied
// (reported as "unknown" if empty).
static iree_status_t iree_hal_coralnpu_stats_create(
    void *self, iree_string_view_t backend_name,
    iree_allocator_t host_allocator, void **out_context) {
  const iree_hal_coralnpu_stats_writer_options_t *options =
      (const iree_hal_coralnpu_stats_writer_options_t *)self;
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(out_context);
  *out_context = NULL;
  if (options->format == IREE_HAL_CORALNPU_STATS_FORMAT_NONE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "CoralNPU stats writer requires an output "
                            "format");
  }
  if (iree_string_view_is_empty(backend_name)) {
    backend_name = IREE_SV("unknown");
  }

  FILE *file = NULL;
  bool owns_file = false;
  IREE_RETURN_IF_ERROR(iree_hal_coralnpu_stats_open_file(
      options->file_path, host_allocator, &file, &owns_file));

  iree_hal_coralnpu_stats_t *stats = NULL;
  iree_status_t status = iree_allocator_malloc_with_trailing(
      host_allocator, sizeof(*stats), backend_name.size, (void **)&stats);
  if (!iree_status_is_ok(status)) {
    if (owns_file) fclose(file);
    return status;
  }
  stats->host_allocator = host_allocator;
  stats->format = options->format;
  stats->collect_fields = options->collect_fields;
  iree_string_view_append_to_buffer(backend_name, &stats->backend_name,
                                    (char *)(stats + 1));
  stats->file = file;
  stats->owns_file = owns_file;
  iree_slim_mutex_initialize(&stats->mutex);
  *out_context = stats;
  return iree_ok_status();
}

// Writes |value| as a JSON string literal.
static void iree_hal_coralnpu_stats_write_json_string(
    FILE *file, iree_string_view_t value) {
  fputc('"', file);
  // Escape quotes, backslashes and control characters, as JSON requires.
  for (iree_host_size_t i = 0; i < value.size; ++i) {
    const unsigned char byte = (unsigned char)value.data[i];
    if (byte == '"' || byte == '\\') {
      fputc('\\', file);
      fputc(byte, file);
      continue;
    }
    if (byte < 0x20) {
      fprintf(file, "\\u%04x", byte);
      continue;
    }
    fputc(byte, file);
  }
  fputc('"', file);
}

// Writes `"<name>":{"start":N,"size":N,"used":N}` for a memory region.
static void iree_hal_coralnpu_stats_write_json_region(
    FILE *file, const char *name,
    const iree_hal_coralnpu_memory_region_t *region) {
  fprintf(file,
          "\"%s\":{\"start\":%" PRIu32 ",\"size\":%" PRIu32 ",\"used\":%" PRIu32
          "}",
          name, region->start, region->size, region->used);
}

// Writes the `memory` object of a JSON record; |has_stack_peak| tells whether
// the stack peak was measured.
static void iree_hal_coralnpu_stats_write_json_memory(
    FILE *file, const iree_hal_coralnpu_memory_stats_t *memory,
    bool has_stack_peak) {
  fputs(",\"memory\":{", file);
  iree_hal_coralnpu_stats_write_json_region(file, "itcm", &memory->itcm);
  fputc(',', file);
  iree_hal_coralnpu_stats_write_json_region(file, "dtcm", &memory->dtcm);
  fputc(',', file);
  iree_hal_coralnpu_stats_write_json_region(file, "heap", &memory->heap);
  fprintf(file, ",\"stack\":{\"start\":%" PRIu32 ",\"size\":%" PRIu32,
          memory->stack.start, memory->stack.size);
  // The stack peak is opt-in.
  if (has_stack_peak) {
    fprintf(file, ",\"peak\":%" PRIu32 ",\"overflow\":%s", memory->stack.used,
            memory->stack_overflow ? "true" : "false");
  }
  fputs("}}", file);
}

// Writes one JSON record (a single line) describing a dispatch.
static void iree_hal_coralnpu_stats_write_json_record(
    iree_hal_coralnpu_stats_t *stats,
    const iree_hal_coralnpu_dispatch_info_t *info,
    const iree_hal_coralnpu_dispatch_stats_t *dispatch_stats,
    iree_status_code_t status_code) {
  FILE *file = stats->file;
  fputs("{\"schema\":\"" IREE_HAL_CORALNPU_STATS_SCHEMA "\"", file);
  fputs(",\"dispatch\":{\"executable\":", file);
  iree_hal_coralnpu_stats_write_json_string(file, info->executable_name);
  fputs(",\"name\":", file);
  iree_hal_coralnpu_stats_write_json_string(file, info->export_name);
  fprintf(file, ",\"ordinal\":%" PRIu32 "}", info->export_ordinal);
  fputs(",\"backend\":{\"simulator\":", file);
  iree_hal_coralnpu_stats_write_json_string(file, stats->backend_name);
  fputc('}', file);
  fprintf(file, ",\"status\":{\"code\":\"%s\"}",
          iree_status_code_string(status_code));
  const uint32_t valid = dispatch_stats->valid_fields;
  const bool has_cycles_total = iree_all_bits_set(
      valid, IREE_HAL_CORALNPU_DISPATCH_STATS_FIELD_CYCLES_TOTAL);
  const bool has_cycles_kernel = iree_all_bits_set(
      valid, IREE_HAL_CORALNPU_DISPATCH_STATS_FIELD_CYCLES_KERNEL);
  // The "cycles" object has only the counts that were reported.
  if (has_cycles_total || has_cycles_kernel) {
    fputs(",\"cycles\":{", file);
    if (has_cycles_total) {
      fprintf(file, "\"total\":%" PRIu64 "%s", dispatch_stats->cycles_total,
              has_cycles_kernel ? "," : "");
    }
    if (has_cycles_kernel) {
      fprintf(file, "\"kernel\":%" PRIu64, dispatch_stats->cycles_kernel);
    }
    fputc('}', file);
  }
  if (iree_all_bits_set(
          valid, IREE_HAL_CORALNPU_DISPATCH_STATS_FIELD_INSTRET_KERNEL)) {
    fprintf(file, ",\"instret\":{\"kernel\":%" PRIu64 "}",
            dispatch_stats->instret_kernel);
  }
  if (iree_all_bits_set(valid, IREE_HAL_CORALNPU_DISPATCH_STATS_FIELD_MEMORY)) {
    iree_hal_coralnpu_stats_write_json_memory(
        file, &dispatch_stats->memory,
        iree_all_bits_set(valid,
                          IREE_HAL_CORALNPU_DISPATCH_STATS_FIELD_STACK_PEAK));
  }
  fputs("}\n", file);
  fflush(file);
}

// Returns the entry aggregating dispatches of |info|, creating it if needed.
static iree_status_t iree_hal_coralnpu_stats_find_or_add_entry(
    iree_hal_coralnpu_stats_t *stats,
    const iree_hal_coralnpu_dispatch_info_t *info,
    iree_hal_coralnpu_stats_entry_t **out_entry) {
  // Use the existing entry of the export, if there is one.
  for (iree_host_size_t i = 0; i < stats->entry_count; ++i) {
    iree_hal_coralnpu_stats_entry_t *entry = stats->entries[i];
    if (entry->export_ordinal == info->export_ordinal &&
        iree_string_view_equal(entry->export_name, info->export_name) &&
        iree_string_view_equal(entry->executable_name, info->executable_name)) {
      *out_entry = entry;
      return iree_ok_status();
    }
  }

  if (stats->entry_count == stats->entry_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        stats->host_allocator, /*minimum_capacity=*/8,
        sizeof(stats->entries[0]), &stats->entry_capacity,
        (void **)&stats->entries));
  }

  iree_hal_coralnpu_stats_entry_t *entry = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_with_trailing(
      stats->host_allocator, sizeof(*entry),
      info->executable_name.size + info->export_name.size, (void **)&entry));
  char *name_storage = (char *)(entry + 1);
  name_storage += iree_string_view_append_to_buffer(
      info->executable_name, &entry->executable_name, name_storage);
  iree_string_view_append_to_buffer(info->export_name, &entry->export_name,
                                    name_storage);
  entry->export_ordinal = info->export_ordinal;
  entry->cycles_min = UINT64_MAX;
  stats->entries[stats->entry_count++] = entry;
  *out_entry = entry;
  return iree_ok_status();
}

// Records the outcome of one dispatch: |status_code| is the code of the status
// the dispatch returned and |dispatch_stats| what the collector collected.
static iree_status_t iree_hal_coralnpu_stats_record(
    iree_hal_coralnpu_stats_t *stats,
    const iree_hal_coralnpu_dispatch_info_t *info,
    const iree_hal_coralnpu_dispatch_stats_t *dispatch_stats,
    iree_status_code_t status_code) {
  IREE_ASSERT_ARGUMENT(stats);
  IREE_ASSERT_ARGUMENT(info);
  IREE_ASSERT_ARGUMENT(dispatch_stats);
  iree_slim_mutex_lock(&stats->mutex);

  // JSON writes one record per dispatch now; pretty aggregates per export
  // and iree_hal_coralnpu_stats_destroy writes the table.
  if (stats->format == IREE_HAL_CORALNPU_STATS_FORMAT_JSON) {
    iree_hal_coralnpu_stats_write_json_record(stats, info, dispatch_stats,
                                              status_code);
    iree_slim_mutex_unlock(&stats->mutex);
    return iree_ok_status();
  }

  iree_hal_coralnpu_stats_entry_t *entry = NULL;
  iree_status_t status =
      iree_hal_coralnpu_stats_find_or_add_entry(stats, info, &entry);
  // Aggregate the dispatch, unless adding its entry failed. The optional
  // statistics count only the dispatches that reported them.
  if (iree_status_is_ok(status)) {
    ++entry->dispatch_count;
    if (status_code != IREE_STATUS_OK) ++entry->failure_count;
    if (iree_all_bits_set(
            dispatch_stats->valid_fields,
            IREE_HAL_CORALNPU_DISPATCH_STATS_FIELD_CYCLES_TOTAL)) {
      const uint64_t cycles = dispatch_stats->cycles_total;
      ++entry->cycles_count;
      entry->cycles_sum += cycles;
      if (cycles < entry->cycles_min) entry->cycles_min = cycles;
      if (cycles > entry->cycles_max) entry->cycles_max = cycles;
    }
    if (iree_all_bits_set(
            dispatch_stats->valid_fields,
            IREE_HAL_CORALNPU_DISPATCH_STATS_FIELD_CYCLES_KERNEL)) {
      ++entry->kernel_cycles_count;
      entry->kernel_cycles_sum += dispatch_stats->cycles_kernel;
    }
    if (iree_all_bits_set(
            dispatch_stats->valid_fields,
            IREE_HAL_CORALNPU_DISPATCH_STATS_FIELD_INSTRET_KERNEL)) {
      ++entry->kernel_instret_count;
      entry->kernel_instret_sum += dispatch_stats->instret_kernel;
    }
    const iree_hal_coralnpu_memory_stats_t *memory = &dispatch_stats->memory;
    if (iree_all_bits_set(dispatch_stats->valid_fields,
                          IREE_HAL_CORALNPU_DISPATCH_STATS_FIELD_MEMORY)) {
      ++entry->memory_count;
      entry->itcm_used_max = iree_max(entry->itcm_used_max, memory->itcm.used);
      entry->dtcm_used_max = iree_max(entry->dtcm_used_max, memory->dtcm.used);
      entry->heap_used_max = iree_max(entry->heap_used_max, memory->heap.used);
    }
    if (iree_all_bits_set(dispatch_stats->valid_fields,
                          IREE_HAL_CORALNPU_DISPATCH_STATS_FIELD_STACK_PEAK)) {
      ++entry->stack_peak_count;
      entry->stack_peak_max =
          iree_max(entry->stack_peak_max, memory->stack.used);
      entry->stack_overflow = entry->stack_overflow || memory->stack_overflow;
    }
  }

  iree_slim_mutex_unlock(&stats->mutex);
  return status;
}

// Looks up the memory layout of |image| in |stats|, reading and caching it on
// first use. Sets |*out_has_memory| to whether |image| has a valid layout and,
// if so, |*out_memory| to it.
static iree_status_t iree_hal_coralnpu_stats_find_or_read_memory(
    iree_hal_coralnpu_stats_t *stats, iree_const_byte_span_t image,
    iree_hal_coralnpu_memory_stats_t *out_memory, bool *out_has_memory) {
  iree_slim_mutex_lock(&stats->mutex);
  for (iree_host_size_t i = 0; i < stats->image_count; ++i) {
    const iree_hal_coralnpu_stats_image_t *cached = &stats->images[i];
    if (cached->image.data == image.data &&
        cached->image.data_length == image.data_length) {
      *out_memory = cached->memory;
      *out_has_memory = cached->has_memory;
      iree_slim_mutex_unlock(&stats->mutex);
      return iree_ok_status();
    }
  }
  if (stats->image_count == stats->image_capacity) {
    iree_status_t status = iree_allocator_grow_array(
        stats->host_allocator, /*minimum_capacity=*/4, sizeof(stats->images[0]),
        &stats->image_capacity, (void **)&stats->images);
    if (!iree_status_is_ok(status)) {
      iree_slim_mutex_unlock(&stats->mutex);
      return status;
    }
  }
  iree_hal_coralnpu_stats_image_t *cached =
      &stats->images[stats->image_count++];
  cached->image = image;
  cached->has_memory =
      iree_hal_coralnpu_firmware_memory_read(image, &cached->memory);
  *out_memory = cached->memory;
  *out_has_memory = cached->has_memory;
  iree_slim_mutex_unlock(&stats->mutex);
  return iree_ok_status();
}

// Starts observing a dispatch: sets |out_hooks| to the hooks of a new
// collector of its statistics, which end_dispatch records and frees.
static iree_status_t iree_hal_coralnpu_stats_begin_dispatch(
    void *self, void *context, const iree_hal_coralnpu_dispatch_info_t *info,
    iree_hal_coralnpu_run_hooks_t *out_hooks) {
  (void)self;
  iree_hal_coralnpu_stats_t *stats = (iree_hal_coralnpu_stats_t *)context;
  IREE_ASSERT_ARGUMENT(stats);
  IREE_ASSERT_ARGUMENT(info);
  IREE_ASSERT_ARGUMENT(out_hooks);
  iree_hal_coralnpu_memory_stats_t memory;
  bool has_memory = false;
  IREE_RETURN_IF_ERROR(iree_hal_coralnpu_stats_find_or_read_memory(
      stats, info->dispatch_image, &memory, &has_memory));
  iree_hal_coralnpu_stats_collector_t *collector = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      stats->host_allocator, sizeof(*collector), (void **)&collector));
  iree_hal_coralnpu_stats_collector_initialize(
      stats->collect_fields, has_memory ? &memory : NULL, collector);
  *out_hooks = iree_hal_coralnpu_stats_collector_hooks(collector);
  return iree_ok_status();
}

// Records the statistics that the collector of |hooks| collected for a
// dispatch that returned a status with |status_code|, and frees the
// collector.
static iree_status_t iree_hal_coralnpu_stats_end_dispatch(
    void *self, void *context, const iree_hal_coralnpu_dispatch_info_t *info,
    const iree_hal_coralnpu_run_hooks_t *hooks,
    iree_status_code_t status_code) {
  (void)self;
  iree_hal_coralnpu_stats_t *stats = (iree_hal_coralnpu_stats_t *)context;
  IREE_ASSERT_ARGUMENT(stats);
  IREE_ASSERT_ARGUMENT(hooks);
  iree_hal_coralnpu_stats_collector_t *collector =
      (iree_hal_coralnpu_stats_collector_t *)hooks->self;
  iree_status_t status = iree_hal_coralnpu_stats_record(
      stats, info, &collector->stats, status_code);
  iree_allocator_free(stats->host_allocator, collector);
  return status;
}

// Writes |value| right-aligned in a |width| column, or "-" if |valid| is
// false.
static void iree_hal_coralnpu_stats_write_column(FILE *file, bool valid,
                                                 uint64_t value, int width) {
  if (!valid) {
    fprintf(file, " %*s", width, "-");
    return;
  }
  fprintf(file, " %*" PRIu64, width, value);
}

// Width of the stack column.
#define IREE_HAL_CORALNPU_STATS_STACK_WIDTH 10

// Writes the peak stack use, followed by "!" if the stack overflowed, or "-"
// if |valid| is false.
static void iree_hal_coralnpu_stats_write_stack_column(FILE *file, bool valid,
                                                       uint32_t peak,
                                                       bool overflow) {
  iree_hal_coralnpu_stats_write_column(
      file, valid, peak,
      IREE_HAL_CORALNPU_STATS_STACK_WIDTH - (valid && overflow ? 1 : 0));
  if (valid && overflow) fputc('!', file);
}

// Writes the summary table of the pretty format.
static void iree_hal_coralnpu_stats_write_summary(
    iree_hal_coralnpu_stats_t *stats) {
  FILE *file = stats->file;
  fprintf(file, "CoralNPU dispatch stats (backend: %.*s)\n",
          (int)stats->backend_name.size, stats->backend_name.data);
  if (stats->entry_count == 0) {
    fputs("  (no dispatches)\n", file);
    fflush(file);
    return;
  }

  fprintf(file,
          " %10s %7s %15s %13s %13s %13s %15s %15s"
          " %9s %9s %9s %*s  %s\n",
          "dispatches", "failed", "cycles(total)", "cycles(avg)", "cycles(min)",
          "cycles(max)", "kcycles(total)", "kinstret(total)", "itcm(max)",
          "data(max)", "heap(max)", IREE_HAL_CORALNPU_STATS_STACK_WIDTH,
          "stack(max)", "executable::export");
  uint64_t total_dispatches = 0;
  uint64_t total_failures = 0;
  uint64_t total_cycles = 0;
  uint64_t total_kernel_cycles = 0;
  uint64_t total_kernel_instret = 0;
  bool all_have_cycles = true;
  bool all_have_kernel_cycles = true;
  bool all_have_kernel_instret = true;
  bool any_has_memory = false;
  uint32_t all_itcm_used_max = 0;
  uint32_t all_dtcm_used_max = 0;
  uint32_t all_heap_used_max = 0;
  bool any_has_stack_peak = false;
  uint32_t all_stack_peak_max = 0;
  bool any_stack_overflow = false;
  for (iree_host_size_t i = 0; i < stats->entry_count; ++i) {
    const iree_hal_coralnpu_stats_entry_t *entry = stats->entries[i];
    // Only report a sum when every dispatch of the entry reported the value;
    // partial sums would be misleading.
    const bool has_cycles = entry->cycles_count == entry->dispatch_count;
    const bool has_kernel_cycles =
        entry->kernel_cycles_count == entry->dispatch_count;
    const bool has_kernel_instret =
        entry->kernel_instret_count == entry->dispatch_count;
    // Unlike sums, maxima of some dispatches are meaningful.
    const bool has_memory = entry->memory_count != 0;
    const bool has_stack_peak = entry->stack_peak_count != 0;
    iree_hal_coralnpu_stats_write_column(file, /*valid=*/true,
                                         entry->dispatch_count, /*width=*/10);
    iree_hal_coralnpu_stats_write_column(file, /*valid=*/true,
                                         entry->failure_count, /*width=*/7);
    iree_hal_coralnpu_stats_write_column(file, has_cycles, entry->cycles_sum,
                                         /*width=*/15);
    iree_hal_coralnpu_stats_write_column(
        file, has_cycles,
        has_cycles ? entry->cycles_sum / entry->dispatch_count : 0,
        /*width=*/13);
    iree_hal_coralnpu_stats_write_column(file, has_cycles, entry->cycles_min,
                                         /*width=*/13);
    iree_hal_coralnpu_stats_write_column(file, has_cycles, entry->cycles_max,
                                         /*width=*/13);
    iree_hal_coralnpu_stats_write_column(file, has_kernel_cycles,
                                         entry->kernel_cycles_sum,
                                         /*width=*/15);
    iree_hal_coralnpu_stats_write_column(file, has_kernel_instret,
                                         entry->kernel_instret_sum,
                                         /*width=*/15);
    iree_hal_coralnpu_stats_write_column(file, has_memory, entry->itcm_used_max,
                                         /*width=*/9);
    iree_hal_coralnpu_stats_write_column(file, has_memory, entry->dtcm_used_max,
                                         /*width=*/9);
    iree_hal_coralnpu_stats_write_column(file, has_memory, entry->heap_used_max,
                                         /*width=*/9);
    iree_hal_coralnpu_stats_write_stack_column(
        file, has_stack_peak, entry->stack_peak_max, entry->stack_overflow);
    fprintf(file, "  %.*s::%.*s\n", (int)entry->executable_name.size,
            entry->executable_name.data, (int)entry->export_name.size,
            entry->export_name.data);
    total_dispatches += entry->dispatch_count;
    total_failures += entry->failure_count;
    total_cycles += entry->cycles_sum;
    total_kernel_cycles += entry->kernel_cycles_sum;
    total_kernel_instret += entry->kernel_instret_sum;
    all_have_cycles = all_have_cycles && has_cycles;
    all_have_kernel_cycles = all_have_kernel_cycles && has_kernel_cycles;
    all_have_kernel_instret = all_have_kernel_instret && has_kernel_instret;
    any_has_memory = any_has_memory || has_memory;
    all_itcm_used_max = iree_max(all_itcm_used_max, entry->itcm_used_max);
    all_dtcm_used_max = iree_max(all_dtcm_used_max, entry->dtcm_used_max);
    all_heap_used_max = iree_max(all_heap_used_max, entry->heap_used_max);
    any_has_stack_peak = any_has_stack_peak || has_stack_peak;
    all_stack_peak_max = iree_max(all_stack_peak_max, entry->stack_peak_max);
    any_stack_overflow = any_stack_overflow || entry->stack_overflow;
  }
  iree_hal_coralnpu_stats_write_column(file, /*valid=*/true, total_dispatches,
                                       /*width=*/10);
  iree_hal_coralnpu_stats_write_column(file, /*valid=*/true, total_failures,
                                       /*width=*/7);
  iree_hal_coralnpu_stats_write_column(file, all_have_cycles, total_cycles,
                                       /*width=*/15);
  // The avg, min and max columns have no total.
  fprintf(file, " %13s %13s %13s", "", "", "");
  iree_hal_coralnpu_stats_write_column(file, all_have_kernel_cycles,
                                       total_kernel_cycles, /*width=*/15);
  iree_hal_coralnpu_stats_write_column(file, all_have_kernel_instret,
                                       total_kernel_instret, /*width=*/15);
  iree_hal_coralnpu_stats_write_column(file, any_has_memory, all_itcm_used_max,
                                       /*width=*/9);
  iree_hal_coralnpu_stats_write_column(file, any_has_memory, all_dtcm_used_max,
                                       /*width=*/9);
  iree_hal_coralnpu_stats_write_column(file, any_has_memory, all_heap_used_max,
                                       /*width=*/9);
  iree_hal_coralnpu_stats_write_stack_column(
      file, any_has_stack_peak, all_stack_peak_max, any_stack_overflow);
  fputs("  (total)\n", file);
  fflush(file);
}

// Destroys the context of a device: writes the summary (for the pretty
// format), closes the output and frees |context|.
static void iree_hal_coralnpu_stats_destroy(void *self,
                                            iree_allocator_t host_allocator,
                                            void *context) {
  (void)self;
  (void)host_allocator;
  iree_hal_coralnpu_stats_t *stats = (iree_hal_coralnpu_stats_t *)context;
  if (!stats) return;
  if (stats->format == IREE_HAL_CORALNPU_STATS_FORMAT_PRETTY) {
    iree_hal_coralnpu_stats_write_summary(stats);
  }
  if (stats->owns_file) {
    fclose(stats->file);
  }
  for (iree_host_size_t i = 0; i < stats->entry_count; ++i) {
    iree_allocator_free(stats->host_allocator, stats->entries[i]);
  }
  iree_allocator_free(stats->host_allocator, stats->entries);
  iree_allocator_free(stats->host_allocator, stats->images);
  iree_slim_mutex_deinitialize(&stats->mutex);
  iree_allocator_free(stats->host_allocator, stats);
}

iree_hal_coralnpu_dispatch_observer_t iree_hal_coralnpu_stats_writer_observer(
    const iree_hal_coralnpu_stats_writer_options_t *options) {
  iree_hal_coralnpu_dispatch_observer_t observer = {
      .self = (void *)options,
      .create = iree_hal_coralnpu_stats_create,
      .destroy = iree_hal_coralnpu_stats_destroy,
      .begin_dispatch = iree_hal_coralnpu_stats_begin_dispatch,
      .end_dispatch = iree_hal_coralnpu_stats_end_dispatch,
  };
  return observer;
}
