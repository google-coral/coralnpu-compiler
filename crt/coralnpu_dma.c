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

// CoralNPU core-local DMA (dma.DmaCore) driver.
//
// Under CORALNPU_DMA_ISSUE_DEFERRED (the default, used in simulation) the
// asynchrony of the engine is modelled here rather than in MPACT, which
// executes the whole transfer synchronously inside the CTRL.START write.
// Starting a transfer only records its configuration, and
// coralnpu_dma_wait_done spins out a modelled beat delay before programming the
// CSRs so premature destination reads observe stale data.
//
// Under CORALNPU_DMA_ISSUE_IMMEDIATE (CoreAxi hardware elaborated with
// Parameters.enableDma = true; default Verilator hw_sim and FPGA builds leave
// enableDma = false, where 0x00032000 is unmapped on-core) start calls program
// the CSRs directly. Both modes enforce single-flight start/wait pairing.

#include "crt/coralnpu_dma.h"

#define REG32(addr) (*(volatile uint32_t *)(uintptr_t)(addr))

// Base STATUS poll floor plus per-16B-beat allowance in
// coralnpu_dma_wait_done(), sized so a 16 MiB (1,048,576-beat) DDR transfer
// has ample headroom before returning CORALNPU_DMA_ETIMEOUT.
#ifndef CORALNPU_DMA_POLL_LIMIT
#define CORALNPU_DMA_POLL_LIMIT 1000000u
#endif  // CORALNPU_DMA_POLL_LIMIT
#ifndef CORALNPU_DMA_POLLS_PER_BEAT
#define CORALNPU_DMA_POLLS_PER_BEAT 64u
#endif  // CORALNPU_DMA_POLLS_PER_BEAT
_Static_assert((uint64_t)CORALNPU_DMA_POLL_LIMIT +
                       1048600ull * (uint64_t)CORALNPU_DMA_POLLS_PER_BEAT <=
                   UINT32_MAX,
               "DMA poll budget overflows uint32_t");

// Modelled cycles per 16-byte beat in deferred issue mode.
#ifndef CORALNPU_DMA_BEAT_CYCLES
#define CORALNPU_DMA_BEAT_CYCLES 4u
#endif  // CORALNPU_DMA_BEAT_CYCLES

enum {
  DMA_STATE_IDLE = 0u,
  DMA_STATE_DEFERRED = 1u,
  DMA_STATE_ISSUED = 2u,
};

typedef struct {
  uint32_t src_addr;
  uint32_t dst_addr;
  uint32_t len_flags;
  uint32_t n_areas;
  uint32_t area_size;
} coralnpu_dma_job_t;

static struct {
  coralnpu_dma_job_t pending_job;
  uint32_t pending_cycles;
  uint32_t state;
  uint32_t seq;
  uint32_t bytes;
  uint32_t beats;
} g_dma;

// Appends one record to the trace region (see coralnpu_dma.h). The runner
// zeroes the count word before the run and reads the region back afterwards.
static void coralnpu_dma_event(uint32_t phase, uint32_t bytes) {
  #if CORALNPU_DMA_TRACE_CAPACITY > 0
    uint32_t n = REG32(CORALNPU_DMA_TRACE_ADDR);
    if (n >= CORALNPU_DMA_TRACE_CAPACITY) return;
    coralnpu_dma_trace_t *rec =
        (coralnpu_dma_trace_t *)(uintptr_t)(CORALNPU_DMA_TRACE_ADDR + 4u +
                                            n * sizeof(coralnpu_dma_trace_t));
    rec->seq = g_dma.seq;
    rec->phase = phase;
    rec->bytes = bytes;
    rec->stamp = coralnpu_dma_stamp();
    REG32(CORALNPU_DMA_TRACE_ADDR) = n + 1u;
  #else
    (void)phase;
    (void)bytes;
  #endif
}

#if CORALNPU_DMA_ISSUE_MODE == CORALNPU_DMA_ISSUE_DEFERRED
// Deterministic delay that needs no CSR support.
static void coralnpu_dma_spin(uint32_t cycles) {
  for (volatile uint32_t i = 0; i < cycles; ++i) { }
}
#endif  // CORALNPU_DMA_ISSUE_MODE

// Programs DmaCore CSRs and triggers START without clearing prior sticky
// errors, preserving RTL Requirement 8 until coralnpu_dma_clear_error().
static void coralnpu_dma_issue(const coralnpu_dma_job_t *job) {
  uint32_t base = coralnpu_dma_base();
  REG32(base + CORALNPU_DMA_SRC_ADDR_OFFSET) = job->src_addr;
  REG32(base + CORALNPU_DMA_DST_ADDR_OFFSET) = job->dst_addr;
  REG32(base + CORALNPU_DMA_LEN_FLAGS_OFFSET) = job->len_flags;
  REG32(base + CORALNPU_DMA_XFER_CFG_OFFSET) = job->n_areas;
  if (job->n_areas > 1u) {
    REG32(base + CORALNPU_DMA_AREA_SIZE_OFFSET) = job->area_size;
  }
  REG32(base + CORALNPU_DMA_CTRL_OFFSET) =
      CORALNPU_DMA_CTRL_ENABLE | CORALNPU_DMA_CTRL_START;
}

int coralnpu_dma_busy(void) {
  if (g_dma.state != DMA_STATE_IDLE) return 1;
  return (REG32(CORALNPU_DMA_STATUS) & CORALNPU_DMA_STATUS_BUSY) != 0u;
}

uint32_t coralnpu_dma_error_flags(void) {
  return REG32(CORALNPU_DMA_STATUS) & CORALNPU_DMA_STATUS_ERROR_FLAGS_MASK;
}

void coralnpu_dma_clear_error(void) {
  uint32_t base = coralnpu_dma_base();
  uint32_t enable =
      REG32(base + CORALNPU_DMA_CTRL_OFFSET) & CORALNPU_DMA_CTRL_ENABLE;
  REG32(base + CORALNPU_DMA_CTRL_OFFSET) =
      enable | CORALNPU_DMA_CTRL_CLEAR_ERROR;
  coralnpu_dma_fence();
}

int coralnpu_dma_deinterleave_async(uint32_t dst, uint32_t src,
                                    uint32_t size_bytes,
                                    uint32_t elem_width_log2,
                                    uint32_t n_areas) {
  // DmaCore reads and writes aligned 16-byte beats and drops bits above [23:0].
  if (((src | dst) & CORALNPU_DMA_ALIGN_MASK) != 0u) {
    return CORALNPU_DMA_EINVAL;
  }
  if (size_bytes == 0u || size_bytes > CORALNPU_DMA_LEN_MASK) {
    return CORALNPU_DMA_EINVAL;
  }
  if (elem_width_log2 > CORALNPU_DMA_MAX_WIDTH_LOG2 || n_areas == 0u ||
      n_areas > CORALNPU_DMA_MAX_AREAS) {
    return CORALNPU_DMA_EINVAL;
  }

  uint32_t area_size = size_bytes;
  if (n_areas > 1u) {
    // Strided mode splits interleaved groups of n_areas elements evenly across
    // planes without a hardware divider, so size_bytes must divide cleanly.
    uint32_t group_bytes = n_areas << elem_width_log2;
    if ((size_bytes % group_bytes) != 0u) return CORALNPU_DMA_EINVAL;
    area_size = size_bytes / n_areas;
  }

  // Configuration registers freeze while running and STATUS.done is
  // level-triggered, so refuse overlapping or un-awaited transfers.
  if (coralnpu_dma_busy()) return CORALNPU_DMA_EBUSY;

  coralnpu_dma_job_t job;
  job.src_addr = src;
  job.dst_addr = dst;
  job.len_flags = coralnpu_dma_make_len_flags(size_bytes, elem_width_log2);
  job.n_areas = n_areas;
  job.area_size = area_size;

  // Publish caller stores to source memory before the DMA engine can read it.
  coralnpu_dma_fence();

  uint32_t passes =
      (area_size + CORALNPU_DMA_ALIGN_MASK) / CORALNPU_DMA_BEAT_BYTES;
  uint32_t beats = passes * n_areas;
  ++g_dma.seq;
  g_dma.bytes = size_bytes;
  g_dma.beats = beats;
  coralnpu_dma_event(CORALNPU_DMA_EVENT_START, g_dma.bytes);

#if CORALNPU_DMA_ISSUE_MODE == CORALNPU_DMA_ISSUE_DEFERRED
  g_dma.pending_job = job;
  g_dma.pending_cycles = beats * CORALNPU_DMA_BEAT_CYCLES;
  g_dma.state = DMA_STATE_DEFERRED;
#else
  g_dma.state = DMA_STATE_ISSUED;
  coralnpu_dma_issue(&job);
#endif  // CORALNPU_DMA_ISSUE_MODE
  return CORALNPU_DMA_OK;
}

int coralnpu_dma_copy_async(uint32_t dst, uint32_t src, uint32_t size_bytes) {
  return coralnpu_dma_deinterleave_async(
      dst, src, size_bytes, /*elem_width_log2=*/0u, /*n_areas=*/1u);
}

int coralnpu_dma_wait_done(void) {
  if (g_dma.state == DMA_STATE_IDLE) return CORALNPU_DMA_OK;

#if CORALNPU_DMA_ISSUE_MODE == CORALNPU_DMA_ISSUE_DEFERRED
  if (g_dma.state == DMA_STATE_DEFERRED) {
    coralnpu_dma_spin(g_dma.pending_cycles);
    coralnpu_dma_fence();
    g_dma.state = DMA_STATE_ISSUED;
    coralnpu_dma_issue(&g_dma.pending_job);
  }
#endif  // CORALNPU_DMA_ISSUE_MODE

  // Wait until the engine returns to idle (BUSY == 0) with either DONE or
  // ERROR set; on xfer_error, ERROR rises before in-flight bus beats finish
  // draining (DmaCsr.scala).
  uint32_t poll_limit =
      CORALNPU_DMA_POLL_LIMIT + g_dma.beats * CORALNPU_DMA_POLLS_PER_BEAT;
  uint32_t status_addr = CORALNPU_DMA_STATUS;
  for (uint32_t i = 0; i < poll_limit; ++i) {
    uint32_t status = REG32(status_addr);
    if ((status & CORALNPU_DMA_STATUS_BUSY) == 0u &&
        (status & (CORALNPU_DMA_STATUS_DONE | CORALNPU_DMA_STATUS_ERROR)) !=
            0u) {
      g_dma.state = DMA_STATE_IDLE;
      coralnpu_dma_fence();
      coralnpu_dma_event(CORALNPU_DMA_EVENT_DONE, g_dma.bytes);
      return (status & CORALNPU_DMA_STATUS_ERROR) ? CORALNPU_DMA_EHARDWARE
                                                  : CORALNPU_DMA_OK;
    }
  }
  return CORALNPU_DMA_ETIMEOUT;
}
