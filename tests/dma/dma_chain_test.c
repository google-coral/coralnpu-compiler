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

// Strided 2D gather and hardware de-interleave DMA test.
//
// Part 1 gathers ROWS rows (ROW_BYTES each, spaced SRC_PITCH bytes apart in
// DDR) into a contiguous DTCM buffer using one 1D transfer per row and checks
// a guard beat past the tile.
//
// Part 2 exercises DmaCore's hardware strided de-interleave mode
// (coralnpu_dma_deinterleave_async) across all valid element widths
// (elem_width_log2 = 0..4, i.e. 1, 2, 4, 8, and 16-byte elements), verifying
// plane padding gaps, shape validation, memcopy ignoring non-zero xfer_width,
// and clean reset back to contiguous memcopy mode.

#include <stdint.h>

#include "crt/coralnpu_dma.h"
#include "tests/dma/dma_test_common.h"

#define ROWS 8u
#define ROW_BYTES 64u  // 16 u32 words per row
#define ROW_WORDS (ROW_BYTES / 4u)
#define SRC_PITCH 256u  // byte stride between source rows in DDR
#define SRC_WORDS ((SRC_PITCH * ROWS) / 4u)
#define GUARD_WORDS (CORALNPU_DMA_BEAT_BYTES / 4u)
#define GUARD_VALUE 0xDEADBEEFu

#define DEINT_AREAS 3u
#define DEINT_MAX_ROW_PITCH 48u
#define DEINT_DST_CAP \
  (DEINT_AREAS * DEINT_MAX_ROW_PITCH + CORALNPU_DMA_BEAT_BYTES)
#define DEINT_SRC_CAP (DEINT_AREAS * DEINT_MAX_ROW_PITCH)
#define DEINT_CANARY 0xCCu

static uint32_t dtcm_tile[ROWS * ROW_WORDS + GUARD_WORDS]
    __attribute__((aligned(64)));
static uint8_t dtcm_planes[DEINT_DST_CAP] __attribute__((aligned(64)));

_Static_assert(ROW_BYTES % CORALNPU_DMA_BEAT_BYTES == 0u,
               "ROW_BYTES must be 16-byte aligned");
_Static_assert(SRC_PITCH % CORALNPU_DMA_BEAT_BYTES == 0u,
               "SRC_PITCH must be 16-byte aligned");
_Static_assert(DDR_SRC % CORALNPU_DMA_BEAT_BYTES == 0u,
               "DDR_SRC must be 16-byte aligned");

static volatile uint32_t *const ddr_src = (volatile uint32_t *)DDR_SRC;
static volatile uint8_t *const ddr_src_u8 = (volatile uint8_t *)DDR_SRC;

// Deterministic byte value for (elem_log2, group_idx, plane, byte_in_elem).
static uint8_t expected_deint_byte(uint32_t elem_log2, uint32_t idx,
                                   uint32_t plane, uint32_t byte_off) {
  return (uint8_t)((idx * 17u + plane * 53u + byte_off * 7u + elem_log2 * 29u) ^
                   0xA5u);
}

// Verifies 3-plane strided de-interleave for one element width (0..4),
// checking payload bytes, row_pitch padding gap bytes, and trailing guard beat.
static int check_deinterleave_width(uint32_t elem_log2,
                                    uint32_t elems_per_area) {
  uint32_t elem_bytes = 1u << elem_log2;
  uint32_t area_bytes = elems_per_area * elem_bytes;
  uint32_t total_bytes = DEINT_AREAS * area_bytes;
  uint32_t row_pitch = coralnpu_dma_row_pitch(area_bytes);
  if (row_pitch != DEINT_MAX_ROW_PITCH) return 10;

  // Populate payload plus final-pass over-read padding up to n_areas *
  // row_pitch bytes (coralnpu_dma.h / dma.md §3 rule 4). Over-read bytes
  // are poisoned with 0xFF so any leak into output planes fails comparison.
  for (uint32_t b = 0; b < DEINT_SRC_CAP; ++b) ddr_src_u8[b] = 0xFFu;
  for (uint32_t idx = 0; idx < elems_per_area; ++idx) {
    for (uint32_t plane = 0; plane < DEINT_AREAS; ++plane) {
      uint32_t base = (idx * DEINT_AREAS + plane) * elem_bytes;
      for (uint32_t b = 0; b < elem_bytes; ++b) {
        ddr_src_u8[base + b] = expected_deint_byte(elem_log2, idx, plane, b);
      }
    }
  }
  for (uint32_t b = 0; b < DEINT_DST_CAP; ++b) dtcm_planes[b] = DEINT_CANARY;

  if (coralnpu_dma_deinterleave_async((uint32_t)(uintptr_t)dtcm_planes, DDR_SRC,
                                      total_bytes, elem_log2,
                                      DEINT_AREAS) != CORALNPU_DMA_OK) {
    return 11;
  }
  if (coralnpu_dma_wait_done() != CORALNPU_DMA_OK) {
    dma_test_result[1] = coralnpu_dma_error_flags();
    return 12;
  }

  for (uint32_t plane = 0; plane < DEINT_AREAS; ++plane) {
    const uint8_t *row = dtcm_planes + plane * row_pitch;
    for (uint32_t idx = 0; idx < elems_per_area; ++idx) {
      for (uint32_t b = 0; b < elem_bytes; ++b) {
        if (row[idx * elem_bytes + b] !=
            expected_deint_byte(elem_log2, idx, plane, b)) {
          return 13;
        }
      }
    }
    for (uint32_t b = area_bytes; b < row_pitch; ++b) {
      if (row[b] != DEINT_CANARY) return 14;
    }
  }
  for (uint32_t b = DEINT_AREAS * row_pitch; b < DEINT_DST_CAP; ++b) {
    if (dtcm_planes[b] != DEINT_CANARY) return 15;
  }
  return 0;
}

// Runs the 2D row-gather and strided de-interleave checks; returns 0 or step.
static int run(void) {
  if (coralnpu_dma_row_pitch(40u) != 48u) return 1;

  for (uint32_t i = 0; i < SRC_WORDS; ++i) ddr_src[i] = i;
  for (uint32_t i = 0; i < ROWS * ROW_WORDS + GUARD_WORDS; ++i) {
    dtcm_tile[i] = GUARD_VALUE;
  }

  // Part 1: per-row 2D strided gather into contiguous DTCM.
  for (uint32_t r = 0; r < ROWS; ++r) {
    uint32_t row_dst = (uint32_t)(uintptr_t)&dtcm_tile[r * ROW_WORDS];
    if (coralnpu_dma_copy_async(row_dst, DDR_SRC + r * SRC_PITCH, ROW_BYTES) !=
        CORALNPU_DMA_OK) {
      return 2;
    }
    if (coralnpu_dma_wait_done() != CORALNPU_DMA_OK) {
      dma_test_result[1] = coralnpu_dma_error_flags();
      return 3;
    }
  }

  for (uint32_t r = 0; r < ROWS; ++r) {
    for (uint32_t c = 0; c < ROW_WORDS; ++c) {
      if (dtcm_tile[r * ROW_WORDS + c] != (r * SRC_PITCH) / 4u + c) return 4;
    }
  }
  for (uint32_t i = 0; i < GUARD_WORDS; ++i) {
    if (dtcm_tile[ROWS * ROW_WORDS + i] != GUARD_VALUE) return 5;
  }

  // Part 2: strided de-interleave validation checks.
  uint32_t planes_addr = (uint32_t)(uintptr_t)dtcm_planes;
  if (coralnpu_dma_deinterleave_async(
          planes_addr, DDR_SRC, /*size_bytes=*/64u,
          /*elem_width_log2=*/CORALNPU_DMA_MAX_WIDTH_LOG2 + 1u,
          /*n_areas=*/2u) != CORALNPU_DMA_EINVAL) {
    return 6;
  }
  if (coralnpu_dma_deinterleave_async(planes_addr, DDR_SRC, /*size_bytes=*/64u,
                                      /*elem_width_log2=*/0u,
                                      /*n_areas=*/0u) != CORALNPU_DMA_EINVAL) {
    return 7;
  }
  // 68 = 4 * 17 divides n_areas=17 cleanly so only MAX_AREAS rejects it.
  if (coralnpu_dma_deinterleave_async(
          planes_addr, DDR_SRC, /*size_bytes=*/68u, /*elem_width_log2=*/0u,
          /*n_areas=*/CORALNPU_DMA_MAX_AREAS + 1u) != CORALNPU_DMA_EINVAL) {
    return 8;
  }
  // 12 bytes divides n_areas=2, so only the full (n_areas << elem_width_log2)
  // = 8-byte group check rejects it.
  if (coralnpu_dma_deinterleave_async(planes_addr, DDR_SRC, /*size_bytes=*/12u,
                                      /*elem_width_log2=*/2u,
                                      /*n_areas=*/2u) != CORALNPU_DMA_EINVAL) {
    return 9;
  }

  // Exercise all five element widths: 1 B (37 B/plane), 2 B (40 B/plane),
  // 4 B (36 B/plane), 8 B (40 B/plane), and 16 B (48 B/plane, beat-level).
  const uint32_t elems_for_width[5] = {37u, 20u, 9u, 5u, 3u};
  for (uint32_t w = 0; w <= CORALNPU_DMA_MAX_WIDTH_LOG2; ++w) {
    int rc = check_deinterleave_width(w, elems_for_width[w]);
    if (rc != 0) return rc;
  }

  // Memcopy (n_areas == 1) must ignore non-zero valid xfer_width (1..4) and
  // reset XFER_CFG.n_areas back to 1 (DmaCsr.scala fillMc.logElemSize := gMax),
  // including lengths not divisible by (1 << xfer_width).
  for (uint32_t w = 1u; w <= CORALNPU_DMA_MAX_WIDTH_LOG2; ++w) {
    const uint32_t copy_len = 37u;
    for (uint32_t b = 0; b < 48u; ++b) {
      ddr_src_u8[b] = (uint8_t)(0x30u + w * 13u + b);
      dtcm_planes[b] = DEINT_CANARY;
    }
    if (coralnpu_dma_deinterleave_async(planes_addr, DDR_SRC, copy_len,
                                        /*elem_width_log2=*/w,
                                        /*n_areas=*/1u) != CORALNPU_DMA_OK) {
      return 16;
    }
    if (coralnpu_dma_wait_done() != CORALNPU_DMA_OK) return 17;
    for (uint32_t b = 0; b < copy_len; ++b) {
      if (dtcm_planes[b] != (uint8_t)(0x30u + w * 13u + b)) return 18;
    }
    for (uint32_t b = copy_len; b < 48u; ++b) {
      if (dtcm_planes[b] != DEINT_CANARY) return 19;
    }
  }

  return 0;
}

int main(void) {
  *dma_test_result = RESULT_TAG_CHAIN | (uint32_t)run();
  return 0;
}
