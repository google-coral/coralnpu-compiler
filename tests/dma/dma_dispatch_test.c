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

// Dispatch firmware for dma_dispatch_runner: verifies clean DmaCore CSR reset
// at dispatch entry, DMA-copies binding 0 into binding 1, checks highmem ITCM
// at 0x00032000 when linked for highmem, and leaves dirty CSRs / sticky errors
// (or DONE on highmem) at exit so the next dispatch on the reused simulator
// verifies per-run CSR reset.

#include "crt/coralnpu_dma.h"

// IREE:
#include "iree/hal/local/executable_library.h"

// Standard library:
#include <stddef.h>
#include <stdint.h>

#define REG32(addr) (*(volatile uint32_t*)(uintptr_t)(addr))
#define HIGHMEM_ITCM_SENTINEL 0xCAFE1234u

// Verifies per-run CSR reset and highmem ITCM, copies binding 0 to binding 1,
// and leaves dirty CSR state at exit; returns 0 on success.
static int dma_copy_dispatch(
    const iree_hal_executable_environment_v0_t* environment,
    const iree_hal_executable_dispatch_state_v0_t* dispatch_state,
    const iree_hal_executable_workgroup_state_v0_t* workgroup_state) {
  (void)environment;
  (void)workgroup_state;
  if (dispatch_state->binding_count != 2) return 1;
  const uint32_t size = (uint32_t)dispatch_state->binding_lengths[0];
  if (dispatch_state->binding_lengths[1] != size) return 2;

  // Reused simulator must start every dispatch with all DmaCore CSRs in their
  // hardware reset state regardless of what the previous dispatch left behind.
  if (REG32(CORALNPU_DMA_CTRL) != 0u || REG32(CORALNPU_DMA_STATUS) != 0u ||
      REG32(CORALNPU_DMA_SRC_ADDR) != 0u ||
      REG32(CORALNPU_DMA_DST_ADDR) != 0u ||
      REG32(CORALNPU_DMA_LEN_FLAGS) != 0u ||
      REG32(CORALNPU_DMA_XFER_CFG) != 1u ||
      REG32(CORALNPU_DMA_AREA_SIZE) != 0u) {
    return 5;
  }

  // On highmem dispatches, 0x00032000 is ITCM rather than DmaCore CSR space.
  if (coralnpu_dma_base() == CORALNPU_DMA_HIGHMEM_BASE &&
      REG32(CORALNPU_DMA_DEFAULT_BASE) != HIGHMEM_ITCM_SENTINEL) {
    return 6;
  }

  const uint32_t src = (uint32_t)(uintptr_t)dispatch_state->binding_ptrs[0];
  const uint32_t dst = (uint32_t)(uintptr_t)dispatch_state->binding_ptrs[1];
  if (coralnpu_dma_copy_async(dst, src, size) != CORALNPU_DMA_OK) {
    return 3;
  }
  if (coralnpu_dma_wait_done() != CORALNPU_DMA_OK) return 4;

  // On default dispatches, latch all three sticky error flags plus non-default
  // XFER_CFG and AREA_SIZE; on highmem, leave DONE and address CSRs set from
  // the completed copy so the next dispatch verifies Reset() clears every CSR.
  if (coralnpu_dma_base() != CORALNPU_DMA_HIGHMEM_BASE) {
    REG32(CORALNPU_DMA_SRC_ADDR) = 0u;
    REG32(CORALNPU_DMA_DST_ADDR) = dst;
    REG32(CORALNPU_DMA_LEN_FLAGS) = 16u;
    REG32(CORALNPU_DMA_XFER_CFG) = 1u;
    REG32(CORALNPU_DMA_CTRL) =
        CORALNPU_DMA_CTRL_ENABLE | CORALNPU_DMA_CTRL_START;
    REG32(CORALNPU_DMA_SRC_ADDR) = src + 4u;
    REG32(CORALNPU_DMA_XFER_CFG) = 3u;
    REG32(CORALNPU_DMA_AREA_SIZE) = 0u;
    REG32(CORALNPU_DMA_CTRL) =
        CORALNPU_DMA_CTRL_ENABLE | CORALNPU_DMA_CTRL_START;
    REG32(CORALNPU_DMA_AREA_SIZE) = 16u;
    if (coralnpu_dma_error_flags() != CORALNPU_DMA_STATUS_ERROR_FLAGS_MASK) {
      return 7;
    }
  }

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
