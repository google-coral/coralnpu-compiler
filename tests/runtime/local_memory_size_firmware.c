// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Test executable library for local_memory_size_test.
//
// Writes workgroup_state's local memory info to binding 0:
//   out[0] = local_memory_size
//   out[1] = local_memory address
//   out[2] = last word of local memory after writing a marker

// IREE:
#include "iree/hal/local/executable_library.h"

// Standard library:
#include <stddef.h>
#include <stdint.h>

#define LOCAL_MEMORY_MARKER 0x10CA1u

// Writes local_memory_size, local_memory, and a test marker to binding 0.
static int report_local_memory(
    const iree_hal_executable_environment_v0_t* environment,
    const iree_hal_executable_dispatch_state_v0_t* dispatch_state,
    const iree_hal_executable_workgroup_state_v0_t* workgroup_state) {
  (void)environment;
  if (dispatch_state->binding_count < 1 ||
      dispatch_state->binding_lengths[0] < 3 * sizeof(uint32_t)) {
    return 1;
  }
  volatile uint32_t* out = (volatile uint32_t*)dispatch_state->binding_ptrs[0];
  const uint32_t size = workgroup_state->local_memory_size;
  out[0] = size;
  out[1] = (uint32_t)(uintptr_t)workgroup_state->local_memory;
  out[2] = 0;
  // Verify the entire buffer is zeroed on entry (catching markers left by prior
  // dispatches on a reused simulator) and dirty every byte for subsequent runs.
  if (size > 0 && workgroup_state->local_memory != NULL) {
    volatile uint32_t* words =
        (volatile uint32_t*)workgroup_state->local_memory;
    const uint32_t word_count = size / sizeof(uint32_t);
    for (uint32_t i = 0; i < word_count; ++i) {
      if (words[i] != 0) {
        return 2;
      }
      words[i] = LOCAL_MEMORY_MARKER;
    }
    // Handle any sub-word tail byte-by-byte to avoid unaligned RV32 accesses.
    volatile uint8_t* tail = (volatile uint8_t*)(words + word_count);
    for (uint32_t i = 0; i < size % sizeof(uint32_t); ++i) {
      if (tail[i] != 0) {
        return 2;
      }
      tail[i] = (uint8_t)LOCAL_MEMORY_MARKER;
    }
    if (word_count > 0) {
      out[2] = words[word_count - 1];
    }
  }
  return 0;
}

static const iree_hal_executable_dispatch_v0_t kExports[] = {
    report_local_memory,
};

static const iree_hal_executable_library_header_t kHeader = {
    .version = IREE_HAL_EXECUTABLE_LIBRARY_VERSION_LATEST,
    .name = "local_memory_size_firmware",
    .features = IREE_HAL_EXECUTABLE_LIBRARY_FEATURE_NONE,
    .sanitizer = IREE_HAL_EXECUTABLE_LIBRARY_SANITIZER_NONE,
};

static const iree_hal_executable_library_v0_t kLibrary = {
    .header = &kHeader,
    .exports =
        {
            .count = 1,
            .ptrs = kExports,
        },
};

// Returns the executable library table.
// TODO: Align crt/coralnpu_main.c and this definition with
// iree_hal_executable_library_query_fn_t (double pointer).
const iree_hal_executable_library_header_t* iree_hal_executable_library_query(
    iree_hal_executable_library_version_t max_version,
    const iree_hal_executable_environment_v0_t* environment) {
  (void)environment;
  if (max_version < IREE_HAL_EXECUTABLE_LIBRARY_VERSION_LATEST) return NULL;
  return (const iree_hal_executable_library_header_t*)&kLibrary;
}
