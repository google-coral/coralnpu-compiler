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

#include "metrics/runtime/stack_peak.h"

#include "iree/base/api.h"

// Number of 32-bit words in each chunk used to paint and scan the stack.
#define IREE_HAL_CORALNPU_STACK_CHUNK_WORDS 64u

void iree_hal_coralnpu_stack_paint(
    const iree_hal_coralnpu_core_access_t* core,
    const iree_hal_coralnpu_memory_stats_t* memory, uint32_t heap_cursor,
    iree_hal_coralnpu_stack_paint_t* out_paint) {
  // Painting and scanning work on whole words.
  const uint32_t end = (memory->stack.start + memory->stack.size) & ~3u;
  const uint32_t stack_start = iree_min((memory->stack.start + 3u) & ~3u, end);
  const uint32_t free_start =
      (iree_max(heap_cursor, memory->heap.start) + 3u) & ~3u;
  uint32_t start = stack_start;
  // Only free heap right below the stack can be the guard area.
  if (memory->heap.start + memory->heap.size == memory->stack.start &&
      free_start < stack_start) {
    start -=
        iree_min(IREE_HAL_CORALNPU_STACK_GUARD_SIZE, stack_start - free_start);
  }
  out_paint->start = start;
  out_paint->stack_start = stack_start;
  out_paint->end = end;

  uint32_t paint[IREE_HAL_CORALNPU_STACK_CHUNK_WORDS];
  for (size_t i = 0; i < IREE_ARRAYSIZE(paint); ++i) {
    paint[i] = IREE_HAL_CORALNPU_STACK_PAINT;
  }
  // Repeat the paint buffer over [start, end).
  for (uint32_t address = start; address < end;) {
    const uint32_t chunk = iree_min((uint32_t)sizeof(paint), end - address);
    core->write_mem(core->self, address, paint, chunk);
    address += chunk;
  }
}

void iree_hal_coralnpu_stack_scan(const iree_hal_coralnpu_core_access_t* core,
                                  const iree_hal_coralnpu_stack_paint_t* paint,
                                  uint32_t* out_peak, bool* out_overflow) {
  uint32_t actual[IREE_HAL_CORALNPU_STACK_CHUNK_WORDS];
  // The stack grows down from |end|, so scan up from the bottom and stop at
  // the first overwritten word.
  uint32_t lowest_written = paint->end;
  for (uint32_t address = paint->start;
       address < paint->end && lowest_written == paint->end;) {
    const uint32_t chunk =
        iree_min((uint32_t)sizeof(actual), paint->end - address);
    core->read_mem(core->self, address, actual, chunk);
    for (uint32_t i = 0; i < chunk / sizeof(uint32_t); ++i) {
      if (actual[i] != IREE_HAL_CORALNPU_STACK_PAINT) {
        lowest_written = address + i * sizeof(uint32_t);
        break;
      }
    }
    address += chunk;
  }

  *out_peak = paint->end - lowest_written;
  // Without a guard area, a fully used stack cannot be told apart from an
  // overflow past the bottom of the stack, so report it as an overflow.
  const bool has_guard = paint->start != paint->stack_start;
  *out_overflow = lowest_written < paint->stack_start ||
                  (!has_guard && paint->start != paint->end &&
                   lowest_written == paint->start);
}
