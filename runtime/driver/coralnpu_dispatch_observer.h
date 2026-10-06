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

#ifndef RUNTIME_DRIVER_CORALNPU_DISPATCH_OBSERVER_H_
#define RUNTIME_DRIVER_CORALNPU_DISPATCH_OBSERVER_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "runtime/driver/coralnpu_run_hooks.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Identifies a dispatch to a dispatch observer. The strings and the image are
// borrowed and only need to remain valid for the duration of the call they are
// passed to.
typedef struct iree_hal_coralnpu_dispatch_info_t {
  // Name of the executable (the hal.executable in the compiled program).
  iree_string_view_t executable_name;
  // Name of the dispatched exported function.
  iree_string_view_t export_name;
  // Ordinal of the dispatched exported function within the executable.
  uint32_t export_ordinal;
  // The firmware ELF of the executable.
  iree_const_byte_span_t dispatch_image;
} iree_hal_coralnpu_dispatch_info_t;

// Vtable of an observer of the dispatches of CoralNPU devices (e.g. the
// statistics writer in metrics/runtime/stats_writer.h). A device with an
// observer lets it hook into the run of every dispatch on the execution
// backend (coralnpu_run_hooks.h); a device without one runs the dispatches
// without hooks. The observer has a context per device.
typedef struct iree_hal_coralnpu_dispatch_observer_t {
  // Passed to the functions below; must outlive the devices using the
  // observer.
  void *self;
  // Creates the context of a device whose dispatches run on the execution
  // backend named |backend_name| (e.g. "mpact"). Optional.
  iree_status_t (*create)(void *self, iree_string_view_t backend_name,
                          iree_allocator_t host_allocator, void **out_context);
  // Destroys the context of a device when the device is destroyed. Optional.
  void (*destroy)(void *self, iree_allocator_t host_allocator, void *context);
  // Called before every dispatch. May set |out_hooks|, which the caller
  // zero-initializes, to the hooks the execution backend calls while it runs
  // the dispatch. Its error is printed as a warning and does not fail the
  // dispatch, which then runs without hooks and without end_dispatch. May be
  // called from several threads at once. NULL means there is no observer.
  iree_status_t (*begin_dispatch)(void *self, void *context,
                                  const iree_hal_coralnpu_dispatch_info_t *info,
                                  iree_hal_coralnpu_run_hooks_t *out_hooks);
  // Called after every dispatch whose begin_dispatch succeeded, also failed
  // ones, with the hooks begin_dispatch set and the code of the status the
  // dispatch returned. Its error is printed as a warning and does not fail
  // the dispatch. May be called from several threads at once. Required with
  // begin_dispatch.
  iree_status_t (*end_dispatch)(void *self, void *context,
                                const iree_hal_coralnpu_dispatch_info_t *info,
                                const iree_hal_coralnpu_run_hooks_t *hooks,
                                iree_status_code_t status_code);
} iree_hal_coralnpu_dispatch_observer_t;

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // RUNTIME_DRIVER_CORALNPU_DISPATCH_OBSERVER_H_
