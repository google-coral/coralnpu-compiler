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

// Negative DMA test: the positive test's first transfer with the wait
// deliberately omitted. The destination MUST still hold stale data when it is
// read, and the runner asserts the mismatch count is nonzero.
//
// This is what makes every other DMA result in the tree trustworthy. The MPACT
// emulation completes a transfer synchronously inside the CTRL write, so only
// the CRT's deferred-issue model (crt/coralnpu_dma.c) makes a missing wait
// observable. If this test ever reports zero mismatches, deferred issue has
// been switched off or compiled out.

#include <stdint.h>

#include "crt/coralnpu_dma.h"
#include "tests/dma/dma_test_common.h"

#define TEST_SIZE 64

static uint32_t dtcm_buf[TEST_SIZE] __attribute__((aligned(64)));

static volatile uint32_t *const ddr_src = (volatile uint32_t *)DDR_SRC;

int main(void) {
  for (int i = 0; i < TEST_SIZE; ++i) {
    ddr_src[i] = 0xA5A50000u + (uint32_t)i;
    dtcm_buf[i] = 0;
  }

  coralnpu_dma_copy_async((uint32_t)(uintptr_t)dtcm_buf, DDR_SRC,
                          sizeof(dtcm_buf));
  // NO coralnpu_dma_wait_done() -- deliberately.

  uint32_t mismatches = 0;
  for (int i = 0; i < TEST_SIZE; ++i) {
    if (dtcm_buf[i] != 0xA5A50000u + (uint32_t)i) ++mismatches;
  }

  // Under deferred issue nothing has moved yet, so mismatches == TEST_SIZE.
  *dma_test_result = RESULT_TAG_NOWAIT | mismatches;
  return 0;
}
