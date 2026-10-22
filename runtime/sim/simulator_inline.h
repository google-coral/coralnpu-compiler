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

#ifndef RUNTIME_SIM_SIMULATOR_INLINE_H_
#define RUNTIME_SIM_SIMULATOR_INLINE_H_

#include <stdbool.h>

#include "iree/base/api.h"
#include "iree/hal/local/executable_library.h"
#include "runtime/driver/coralnpu_run_hooks.h"
#include "runtime/sim/simulator_api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Runs export |ordinal| of |dispatch_image| on |sim|. |binding_writeable| has
// one entry per binding. If |local_memory_size| is not 0, the workgroup local
// memory of that size is allocated in the firmware heap (in DTCM) and zeroed,
// and the dispatch request gives its address and size to the firmware. Calls
// |hooks|, unless it is NULL, around the run of the program.
iree_status_t iree_hal_simulator_issue_dispatch_inline(
    coralnpu_simulator_t *sim, iree_const_byte_span_t dispatch_image,
    const iree_hal_executable_dispatch_state_v0_t *dispatch_state,
    const bool *binding_writeable, iree_host_size_t ordinal,
    iree_host_size_t local_memory_size,
    const iree_hal_coralnpu_run_hooks_t *hooks);

// Drops cached state for |sim| before it is destroyed.
void iree_hal_simulator_inline_forget(coralnpu_simulator_t *sim);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // RUNTIME_SIM_SIMULATOR_INLINE_H_
