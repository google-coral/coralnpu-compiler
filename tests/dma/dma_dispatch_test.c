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

// Dispatch firmware for dma_dispatch_runner: DMA-copies binding 0 into
// binding 1, returning the failed step or 0.

#include "crt/coralnpu_dma.h"

// IREE:
#include "iree/hal/local/executable_library.h"

// Standard library:
#include <stddef.h>
#include <stdint.h>

static coralnpu_dma_descriptor_t desc;

// Copies binding 0 to binding 1 via async DMA and returns 0 on success.
static int dma_copy_dispatch(
    const iree_hal_executable_environment_v0_t* environment,
    const iree_hal_executable_dispatch_state_v0_t* dispatch_state,
    const iree_hal_executable_workgroup_state_v0_t* workgroup_state) {
  (void)environment;
  (void)workgroup_state;
  if (dispatch_state->binding_count != 2) return 1;
  const uint32_t size = (uint32_t)dispatch_state->binding_lengths[0];
  if (dispatch_state->binding_lengths[1] != size) return 2;

  const uint32_t src = (uint32_t)(uintptr_t)dispatch_state->binding_ptrs[0];
  const uint32_t dst = (uint32_t)(uintptr_t)dispatch_state->binding_ptrs[1];
  if (coralnpu_dma_copy_async(dst, src, size, &desc) != CORALNPU_DMA_OK) {
    return 3;
  }
  if (coralnpu_dma_wait_done() != CORALNPU_DMA_OK) return 4;
  return 0;
}

static const iree_hal_executable_dispatch_v0_t exports[] = {
    dma_copy_dispatch,
};

static const iree_hal_executable_library_header_t header = {
    .version = IREE_HAL_EXECUTABLE_LIBRARY_VERSION_LATEST,
    .name = "dma_dispatch_test",
    .features = IREE_HAL_EXECUTABLE_LIBRARY_FEATURE_NONE,
    .sanitizer = IREE_HAL_EXECUTABLE_LIBRARY_SANITIZER_NONE,
};

static const iree_hal_executable_library_v0_t library = {
    .header = &header,
    .exports =
        {
            .count = 1,
            .ptrs = exports,
        },
};

// Called by coralnpu_main.c in libcoralnpu_iree.a.
const iree_hal_executable_library_header_t* iree_hal_executable_library_query(
    iree_hal_executable_library_version_t max_version,
    const iree_hal_executable_environment_v0_t* environment) {
  (void)environment;
  if (max_version < IREE_HAL_EXECUTABLE_LIBRARY_VERSION_LATEST) return NULL;
  return (const iree_hal_executable_library_header_t*)&library;
}
