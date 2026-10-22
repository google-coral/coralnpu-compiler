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

#ifndef RUNTIME_DRIVER_CORALNPU_DEVICE_H_
#define RUNTIME_DRIVER_CORALNPU_DEVICE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "runtime/driver/coralnpu_dispatch_observer.h"
#include "runtime/driver/coralnpu_exec_backend.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Parameters configuring an iree_hal_coralnpu_device_t.
// Must be initialized with iree_hal_coralnpu_device_params_initialize prior to
// use.
typedef struct iree_hal_coralnpu_device_params_t {
  // Total size of each block in the device shared block pool.
  // Larger sizes will lower overhead and ensure the heap isn't hit for
  // transient allocations while also increasing memory consumption.
  iree_host_size_t arena_block_size;
  // Observer of the dispatches (e.g. a statistics writer); none by default.
  // The device runs the dispatches with run hooks only when it has an
  // observer.
  iree_hal_coralnpu_dispatch_observer_t dispatch_observer;
} iree_hal_coralnpu_device_params_t;

// Initializes |out_params| to default values.
void iree_hal_coralnpu_device_params_initialize(
    iree_hal_coralnpu_device_params_t *out_params);

// Creates a new synchronous local CoralNPU device that performs execution
// inline on threads issuing submissions. |exec_backend| is required and
// supplies the execution backend vtable.
iree_status_t iree_hal_coralnpu_device_create(
    iree_string_view_t identifier,
    const iree_hal_coralnpu_device_params_t *params,
    const iree_hal_device_create_params_t *create_params,
    const iree_hal_coralnpu_exec_backend_t *exec_backend,
    iree_hal_allocator_t *device_allocator, iree_allocator_t host_allocator,
    iree_hal_device_t **out_device);

// Issues an inline dispatch on the device's execution backend. |info|
// identifies the dispatch to the device's dispatch observer.
iree_status_t iree_hal_coralnpu_device_dispatch(
    iree_hal_device_t *base_device,
    const iree_hal_coralnpu_dispatch_info_t *info,
    iree_const_byte_span_t dispatch_image,
    const iree_hal_executable_dispatch_state_v0_t *dispatch_state,
    const bool *binding_writeable, iree_host_size_t ordinal,
    iree_host_size_t local_memory_size);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // RUNTIME_DRIVER_CORALNPU_DEVICE_H_
