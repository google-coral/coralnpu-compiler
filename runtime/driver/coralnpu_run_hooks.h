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
} iree_hal_coralnpu_core_access_t;

// The run of a dispatch: the core runs the program of the dispatch from its
// entry point until the core stops.
typedef struct iree_hal_coralnpu_run_t {
  // The core that runs the dispatch.
  iree_hal_coralnpu_core_access_t core;
} iree_hal_coralnpu_run_t;

// Hooks that an execution backend calls while it runs a dispatch. Each hook
// may be NULL. The hooks can't fail the dispatch.
typedef struct iree_hal_coralnpu_run_hooks_t {
  // Passed to the hooks below.
  void *self;
  // Called right before the run, once the program and its data are loaded.
  void (*before_run)(void *self, const iree_hal_coralnpu_run_t *run);
  // Called right after the run, also when the core did not halt. Not called
  // when the dispatch fails before the run.
  void (*after_run)(void *self, const iree_hal_coralnpu_run_t *run);
} iree_hal_coralnpu_run_hooks_t;

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // RUNTIME_DRIVER_CORALNPU_RUN_HOOKS_H_
