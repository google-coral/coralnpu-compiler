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

// Measures the peak stack use of a run: paints the stack of the firmware
// before the run and finds the lowest word that the firmware overwrote after
// it. Works on any core through the run hooks (coralnpu_run_hooks.h).

#ifndef METRICS_RUNTIME_STACK_PEAK_H_
#define METRICS_RUNTIME_STACK_PEAK_H_

#include <stdbool.h>
#include <stdint.h>

#include "metrics/runtime/dispatch_stats.h"
#include "runtime/driver/coralnpu_run_hooks.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Word painted over the stack before a run; the words that differ from it
// after the run were written by the firmware.
#define IREE_HAL_CORALNPU_STACK_PAINT 0xDEADBEEFu

// Size of the free heap area right below the stack that is painted too, so
// that an overflow can be told apart from a fully used stack.
#define IREE_HAL_CORALNPU_STACK_GUARD_SIZE 256u

// The memory painted before a run, [start, end): the stack and, below it, the
// guard area [start, stack_start), which may be empty. All are word-aligned.
typedef struct iree_hal_coralnpu_stack_paint_t {
  uint32_t start;
  uint32_t stack_start;
  uint32_t end;
} iree_hal_coralnpu_stack_paint_t;

// Paints the stack in |memory| on |core|, and the guard area below it, which
// is taken from the free heap above |heap_cursor| when the heap ends where
// the stack starts. Sets |out_paint| to the painted memory. Call it once the
// firmware and the dispatch data are loaded.
void iree_hal_coralnpu_stack_paint(
    const iree_hal_coralnpu_core_access_t *core,
    const iree_hal_coralnpu_memory_stats_t *memory, uint32_t heap_cursor,
    iree_hal_coralnpu_stack_paint_t *out_paint);

// Scans |paint| on |core| after the run. Sets |out_peak| to the bytes from
// the end of the stack down to the lowest overwritten word, and
// |out_overflow| to whether the firmware overwrote a guard word or, without a
// guard area, the whole stack. Because frame space that the firmware reserves
// without writing is not seen, |out_peak| is a lower bound and a false
// |out_overflow| means only that no overflow was detected.
void iree_hal_coralnpu_stack_scan(const iree_hal_coralnpu_core_access_t *core,
                                  const iree_hal_coralnpu_stack_paint_t *paint,
                                  uint32_t *out_peak, bool *out_overflow);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // METRICS_RUNTIME_STACK_PEAK_H_
