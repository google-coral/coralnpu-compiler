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

// Hooks through which code outside the HAL driver (e.g. the statistics in
// metrics/runtime) observes how an execution backend (coralnpu_exec_backend.h)
// runs a dispatch, without the backend knowing what the hooks measure.

#ifndef RUNTIME_DRIVER_CORALNPU_RUN_HOOKS_H_
#define RUNTIME_DRIVER_CORALNPU_RUN_HOOKS_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Access to the core that runs a dispatch.
typedef struct iree_hal_coralnpu_core_access_t {
  // Passed to the functions below.
  void *self;
  // Returns the cycle counter of the core; 0 if the backend has none.
  uint64_t (*cycle_count)(void *self);
  // Reads |size| bytes at |address| of the memory of the core into |data|.
  // The read may advance the cycle counter (e.g. on Verilator).
  void (*read_mem)(void *self, uint32_t address, void *data, size_t size);
  // Writes |size| bytes of |data| to |address| of the memory of the core.
  // The write may advance the cycle counter (e.g. on Verilator).
  void (*write_mem)(void *self, uint32_t address, const void *data,
                    size_t size);
} iree_hal_coralnpu_core_access_t;

// The run of a dispatch: the core runs the program of the dispatch from its
// entry point until the core stops.
typedef struct iree_hal_coralnpu_run_t {
  // The core that runs the dispatch.
  iree_hal_coralnpu_core_access_t core;
  // Address of the dispatch request of the firmware
  // (crt/coralnpu_dispatch.h).
  uint32_t dispatch_request_addr;
  // End of the dispatch data that the host placed in the heap of the
  // firmware, which starts at `__heap_start`.
  uint32_t heap_cursor;
  // Whether the core halted; set before after_run.
  bool halted;
} iree_hal_coralnpu_run_t;

// Hooks that an execution backend calls while it runs a dispatch. Each hook
// may be NULL. The hooks can't fail the dispatch.
typedef struct iree_hal_coralnpu_run_hooks_t {
  // Passed to the hooks below.
  void *self;
  // Called right before the run, once the program and its data are loaded.
  // May only write unallocated memory that the firmware does not read before
  // writing (the free heap [heap_cursor, __heap_end) and the stack
  // [__stack_start, __stack_start + __stack_size)).
  void (*before_run)(void *self, const iree_hal_coralnpu_run_t *run);
  // Called right after the run, also when the core did not halt. Not called
  // when the dispatch fails before the run.
  void (*after_run)(void *self, const iree_hal_coralnpu_run_t *run);
} iree_hal_coralnpu_run_hooks_t;

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // RUNTIME_DRIVER_CORALNPU_RUN_HOOKS_H_
