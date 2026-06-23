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

#include <stddef.h>
#include <stdint.h>

#include "crt/coralnpu_dispatch.h"
#include "iree/hal/local/executable_library.h"

extern const iree_hal_executable_library_header_t*
iree_hal_executable_library_query(
    iree_hal_executable_library_version_t max_version,
    const iree_hal_executable_environment_v0_t* environment);

__attribute__((section(".noinit.coralnpu_dispatch_request"), aligned(16), used,
               visibility("default"))) coralnpu_dispatch_request_t
    coralnpu_dispatch_request;

static const iree_hal_executable_environment_v0_t environment = {0};

__attribute__((noreturn)) static void coralnpu_halt(void) {
  __asm__ volatile(".word 0x08000073" ::: "memory");
  __builtin_unreachable();
}

__attribute__((noreturn)) static void coralnpu_dispatch_fail(
    coralnpu_dispatch_request_t* request, int32_t return_code) {
  request->return_code = return_code;
  request->status = CORALNPU_DISPATCH_STATUS_ERROR;
  coralnpu_halt();
}

void main(void) {
  coralnpu_dispatch_request_t* request = &coralnpu_dispatch_request;

  if (request->magic != CORALNPU_DISPATCH_MAGIC ||
      request->version != CORALNPU_DISPATCH_VERSION ||
      request->status != CORALNPU_DISPATCH_STATUS_READY) {
    coralnpu_dispatch_fail(request, -1);
  }

  if ((request->push_constant_count != 0 &&
       request->push_constants_addr == 0) ||
      (request->binding_count != 0 && (request->binding_ptrs_addr == 0 ||
                                       request->binding_lengths_addr == 0))) {
    coralnpu_dispatch_fail(request, -2);
  }

  request->status = CORALNPU_DISPATCH_STATUS_RUNNING;
  request->return_code = 0;

  const iree_hal_executable_library_header_t* library_header =
      iree_hal_executable_library_query(
          IREE_HAL_EXECUTABLE_LIBRARY_VERSION_LATEST, &environment);

  if (library_header == NULL) {
    coralnpu_dispatch_fail(request, -3);
  }

  const iree_hal_executable_library_v0_t* library =
      (const iree_hal_executable_library_v0_t*)library_header;

  if (library->exports.ptrs == NULL ||
      request->ordinal >= library->exports.count ||
      library->exports.ptrs[request->ordinal] == NULL) {
    coralnpu_dispatch_fail(request, -4);
  }

  iree_hal_executable_dispatch_state_v0_t dispatch_state;

  dispatch_state.workgroup_size_x = request->workgroup_size_x;
  dispatch_state.workgroup_size_y = request->workgroup_size_y;
  dispatch_state.workgroup_size_z = request->workgroup_size_z;

  dispatch_state.workgroup_count_x = request->workgroup_count_x;
  dispatch_state.workgroup_count_y = request->workgroup_count_y;
  dispatch_state.workgroup_count_z = request->workgroup_count_z;

  dispatch_state.max_concurrency = request->max_concurrency;
  dispatch_state.constant_count = request->push_constant_count;
  dispatch_state.binding_count = request->binding_count;

  dispatch_state.constants =
      request->push_constant_count == 0
          ? NULL
          : (const uint32_t*)(uintptr_t)request->push_constants_addr;

  dispatch_state.binding_ptrs =
      request->binding_count == 0
          ? NULL
          : (void* const*)(uintptr_t)request->binding_ptrs_addr;

  dispatch_state.binding_lengths =
      request->binding_count == 0
          ? NULL
          : (const size_t*)(uintptr_t)request->binding_lengths_addr;

  iree_hal_executable_dispatch_v0_t dispatch =
      library->exports.ptrs[request->ordinal];

  for (uint32_t z = 0; z < request->workgroup_count_z; ++z) {
    for (uint32_t y = 0; y < request->workgroup_count_y; ++y) {
      for (uint32_t x = 0; x < request->workgroup_count_x; ++x) {
        iree_hal_executable_workgroup_state_v0_t workgroup_state;

        workgroup_state.workgroup_id_x = x;
        workgroup_state.workgroup_id_y = y;
        workgroup_state.workgroup_id_z = z;
        workgroup_state.processor_id = 0;
        workgroup_state.local_memory =
            request->local_memory_addr == 0
                ? NULL
                : (uint8_t*)(uintptr_t)request->local_memory_addr;

        int result = dispatch(&environment, &dispatch_state, &workgroup_state);

        if (result != 0) {
          coralnpu_dispatch_fail(request, result);
        }
      }
    }
  }

  request->return_code = 0;
  request->status = CORALNPU_DISPATCH_STATUS_COMPLETE;
  coralnpu_halt();
}
