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

// Timeout and retry-wait test for crt/coralnpu_dma.c.
//
// Linked with --defsym=__dma_base=0x20001000 so coralnpu_dma_base() points at
// EXTMEM SRAM instead of MPACT's synchronous DmaCore model. Verifies:
//   1. DEFINED(__dma_base) linker override resolves to 0x20001000.
//   2. wait_done() returns CORALNPU_DMA_ETIMEOUT when STATUS stays 0 (and when
//      STATUS has BUSY | ERROR asserted during xfer_error drain).
//   3. Poll budget scales with transfer beat count (65-beat timeout retires
//      at least 64 * 64 more instructions than a 1-beat timeout).
//   4. While timed out (DMA_STATE_ISSUED), busy() stays nonzero, new starts
//      return EBUSY, and a retry wait_done() re-polls STATUS to completion
//      without re-writing CTRL.START.

#include "crt/coralnpu_dma.h"
#include "tests/dma/dma_test_common.h"

// Standard library:
#include <stdint.h>

#define FAKE_DMA_BASE 0x20001000u
#define REG32(addr) (*(volatile uint32_t *)(uintptr_t)(addr))

static uint32_t dtcm_buf[272] __attribute__((aligned(64)));

// Reads retired instruction count for poll-budget scaling verification.
static uint64_t read_minstret(void) {
  uint32_t hi0, lo, hi1;
  do {
    __asm__ volatile("csrr %0, minstreth" : "=r"(hi0));
    __asm__ volatile("csrr %0, minstret" : "=r"(lo));
    __asm__ volatile("csrr %0, minstreth" : "=r"(hi1));
  } while (hi0 != hi1);
  return ((uint64_t)hi0 << 32) | lo;
}

// Exercises timeout, STATUS.BUSY gating, poll scaling, and retry wait_done().
static int run(void) {
  if (coralnpu_dma_base() != FAKE_DMA_BASE) return 1;

  for (uint32_t off = 0; off < 0x20u; off += 4u) {
    REG32(FAKE_DMA_BASE + off) = 0u;
  }

  // 1-beat (16 B) transfer against STATUS == 0 must time out.
  uint32_t dst_addr = (uint32_t)(uintptr_t)dtcm_buf;
  if (coralnpu_dma_copy_async(dst_addr, DDR_SRC, 16u) != CORALNPU_DMA_OK) {
    return 2;
  }
  uint64_t inst_start_1 = read_minstret();
  if (coralnpu_dma_wait_done() != CORALNPU_DMA_ETIMEOUT) return 3;
  uint64_t inst_1beat = read_minstret() - inst_start_1;

  // CTRL must have received ENABLE | START, and driver must remain ISSUED.
  if (REG32(CORALNPU_DMA_CTRL) !=
      (CORALNPU_DMA_CTRL_ENABLE | CORALNPU_DMA_CTRL_START)) {
    return 4;
  }
  if (!coralnpu_dma_busy()) return 5;
  if (coralnpu_dma_copy_async(dst_addr, DDR_SRC, 16u) != CORALNPU_DMA_EBUSY) {
    return 6;
  }

  // While BUSY remains asserted alongside ERROR (in-flight beats still
  // draining), retry wait_done() must still wait and time out without
  // re-issuing CTRL.START.
  REG32(CORALNPU_DMA_CTRL) = 0u;
  REG32(CORALNPU_DMA_STATUS) =
      CORALNPU_DMA_STATUS_BUSY | CORALNPU_DMA_STATUS_ERROR;
  if (coralnpu_dma_wait_done() != CORALNPU_DMA_ETIMEOUT) return 7;
  if (REG32(CORALNPU_DMA_CTRL) != 0u || !coralnpu_dma_busy()) return 8;

  // Once STATUS reports BUSY == 0 and DONE == 1, retry wait_done() must
  // complete cleanly without writing CTRL again.
  REG32(CORALNPU_DMA_STATUS) = CORALNPU_DMA_STATUS_DONE;
  if (coralnpu_dma_wait_done() != CORALNPU_DMA_OK) return 9;
  if (REG32(CORALNPU_DMA_CTRL) != 0u || coralnpu_dma_busy()) return 10;

  // When DMA_STATE_IDLE, hardware STATUS.BUSY alone must still report busy()
  // and refuse a new start with EBUSY.
  REG32(CORALNPU_DMA_STATUS) = CORALNPU_DMA_STATUS_BUSY;
  if (!coralnpu_dma_busy() ||
      coralnpu_dma_copy_async(dst_addr, DDR_SRC, 16u) != CORALNPU_DMA_EBUSY) {
    return 10;
  }

  // 65-beat (1040 B) transfer must scale poll_limit by 64 * 64 = 4096 polls
  // relative to the 1-beat transfer above.
  REG32(CORALNPU_DMA_STATUS) = 0u;
  if (coralnpu_dma_copy_async(dst_addr, DDR_SRC, 65u * 16u) !=
      CORALNPU_DMA_OK) {
    return 11;
  }
  uint64_t inst_start_65 = read_minstret();
  if (coralnpu_dma_wait_done() != CORALNPU_DMA_ETIMEOUT) return 12;
  uint64_t inst_65beat = read_minstret() - inst_start_65;
  if (inst_65beat <= inst_1beat + 64ull * 64ull) return 13;

  // Complete the second transfer via retry wait_done() on EHARDWARE when
  // BUSY clears with ERROR set.
  REG32(CORALNPU_DMA_CTRL) = 0u;
  REG32(CORALNPU_DMA_STATUS) =
      CORALNPU_DMA_STATUS_ERROR | CORALNPU_DMA_STATUS_XFER_ERROR;
  if (coralnpu_dma_wait_done() != CORALNPU_DMA_EHARDWARE) return 14;
  if (REG32(CORALNPU_DMA_CTRL) != 0u || coralnpu_dma_busy()) return 15;

  // Idle wait_done() with STATUS == ERROR | XFER_ERROR must return OK without
  // polling STATUS or emitting a duplicate DONE trace record.
  if (coralnpu_dma_wait_done() != CORALNPU_DMA_OK) return 16;

  // Two transfers issued and completed across retries -> 4 trace records.
  if (REG32(CORALNPU_DMA_TRACE_ADDR) != 4u) return 17;

  return 0;
}

int main(void) {
  *dma_test_result = RESULT_TAG_TIMEOUT | (uint32_t)run();
  return 0;
}
