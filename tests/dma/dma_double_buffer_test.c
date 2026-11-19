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

// Double-buffered (ping-pong) DMA test.
//
// Streams NUM_TILES tiles from DDR into a two-slot DTCM buffer (`dtcm_buf[2]`).
// Each iteration waits for `dtcm_buf[curr]`, starts prefetching the next tile
// into `dtcm_buf[next]`, and accumulates `dtcm_buf[curr]` into `dtcm_accum`.
// Finally, `dtcm_accum` is DMA-copied back to DDR.
//
// In simulation (deferred issue), data only moves inside `wait_done()`, so
// this test checks the ping-pong schedule (buffer indexing and start/wait
// pairing) rather than true hardware overlap. For the same reason it cannot
// catch a prefetch that overwrites the buffer still being read: deferred issue
// delays that write until after the compute, whereas on hardware it would race
// the engine.
//
// Writes RESULT_TAG_DOUBLE_BUF on success, or sets the low 16 bits to the
// failing step number.

#include <stdint.h>

#include "crt/coralnpu_dma.h"
#include "tests/dma/dma_test_common.h"

#define NUM_TILES 4u
#define TILE_WORDS 64u  // 256 bytes per tile (multiple of 16-byte beat)
#define TILE_BYTES (TILE_WORDS * 4u)

static uint32_t dtcm_buf[2][TILE_WORDS] __attribute__((aligned(64)));
static uint32_t dtcm_accum[TILE_WORDS] __attribute__((aligned(64)));

static volatile uint32_t *const ddr_src = (volatile uint32_t *)DDR_SRC;
static volatile uint32_t *const ddr_dst = (volatile uint32_t *)DDR_DST;

static inline uint32_t tile_word(uint32_t tile, uint32_t index) {
  return ((tile + 1u) << 16) + index;
}

static int run(void) {
  for (uint32_t t = 0; t < NUM_TILES; ++t) {
    for (uint32_t i = 0; i < TILE_WORDS; ++i) {
      ddr_src[t * TILE_WORDS + i] = tile_word(t, i);
    }
  }
  for (uint32_t i = 0; i < TILE_WORDS; ++i) {
    dtcm_buf[0][i] = 0;
    dtcm_buf[1][i] = 0;
    dtcm_accum[i] = 0;
    ddr_dst[i] = 0;
  }

  // Prologue: start fetching tile 0 into buffer 0.
  if (coralnpu_dma_copy_async((uint32_t)(uintptr_t)dtcm_buf[0], DDR_SRC,
                              TILE_BYTES) != CORALNPU_DMA_OK) {
    return 1;
  }

  for (uint32_t t = 0; t < NUM_TILES; ++t) {
    uint32_t curr = t & 1u;
    uint32_t next = curr ^ 1u;

    // 1. Wait for dtcm_buf[curr].
    if (coralnpu_dma_wait_done() != CORALNPU_DMA_OK) {
      dma_test_result[1] = coralnpu_dma_error_flags();
      return 2;
    }

    // 2. Start prefetching tile (t + 1) into dtcm_buf[next].
    if (t + 1u < NUM_TILES) {
      uint32_t next_src = DDR_SRC + (t + 1u) * TILE_BYTES;
      if (coralnpu_dma_copy_async((uint32_t)(uintptr_t)dtcm_buf[next], next_src,
                                  TILE_BYTES) != CORALNPU_DMA_OK) {
        return 3;
      }
    }

    // 3. Accumulate dtcm_buf[curr] while dtcm_buf[next] is pending/in flight.
    for (uint32_t i = 0; i < TILE_WORDS; ++i) {
      if (dtcm_buf[curr][i] != tile_word(t, i)) return 4;
      dtcm_accum[i] += dtcm_buf[curr][i];
    }
  }

  // Epilogue: copy dtcm_accum back to DDR_DST.
  if (coralnpu_dma_copy_async(DDR_DST, (uint32_t)(uintptr_t)dtcm_accum,
                              TILE_BYTES) != CORALNPU_DMA_OK) {
    return 5;
  }
  if (coralnpu_dma_wait_done() != CORALNPU_DMA_OK) {
    dma_test_result[1] = coralnpu_dma_error_flags();
    return 6;
  }

  // Sum of tile_word(t, i) = ((t + 1) << 16) + i over t = 0..NUM_TILES-1:
  // (NUM_TILES * (NUM_TILES + 1) / 2) << 16, plus NUM_TILES * i.
  for (uint32_t i = 0; i < TILE_WORDS; ++i) {
    uint32_t expected =
        ((NUM_TILES * (NUM_TILES + 1u) / 2u) << 16) + NUM_TILES * i;
    if (ddr_dst[i] != expected) return 7;
  }

  return 0;
}

int main(void) {
  *dma_test_result = RESULT_TAG_DOUBLE_BUF | (uint32_t)run();
  return 0;
}
