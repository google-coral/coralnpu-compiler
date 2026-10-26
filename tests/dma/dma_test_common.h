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

// Memory layout shared by the standalone DMA guest tests.
//
// DTCM buffers are ordinary statics: coralnpu_tcm.ld places .data/.bss in
// DTCM. DDR has no loadable section the simulator's ELF loader accepts, so DDR
// buffers are fixed absolute addresses at the bottom of DDR (0x80000000). The
// runner reads RESULT_ADDR back after the run, followed by an auxiliary word a
// test may use for a diagnostic value; the DMA trace region
// (CORALNPU_DMA_TRACE_ADDR) sits above both.

#ifndef TESTS_DMA_DMA_TEST_COMMON_H_
#define TESTS_DMA_DMA_TEST_COMMON_H_

#include <stdint.h>

#define DDR_BASE 0x80000000u
#define DDR_SRC (DDR_BASE + 0x0000u)
#define DDR_DST (DDR_BASE + 0x1000u)
#define RESULT_ADDR (DDR_BASE + 0x2000u)

// Result word tags, one per test, so a stray value is attributable.
#define RESULT_TAG_RUNTIME 0xD11A0000u
#define RESULT_TAG_NOWAIT 0xD11B0000u
#define RESULT_TAG_CHAIN 0xD11C0000u
// #define RESULT_TAG_PROBE 0xD11D0000u // TODO: Probe test
#define RESULT_TAG_DOUBLE_BUF 0xD11E0000u

#ifndef __cplusplus
static volatile uint32_t *const dma_test_result =
    (volatile uint32_t *)RESULT_ADDR;
#endif

#endif  // TESTS_DMA_DMA_TEST_COMMON_H_
