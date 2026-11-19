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

// Positive DmaCore test: live CSR aperture check, DDR<->DTCM transfers, partial
// tail-beat masking, single-flight busy guard, alignment/length validation,
// sticky hardware error blocking + recovery, invalid-offset/width CSR access
// faults, and idle wait_done(). Linked under both coralnpu_tcm.ld and
// coralnpu_tcm_highmem.ld with explicit -DEXPECTED_DMA_BASE.

#include <stdint.h>

#include "crt/coralnpu_dma.h"
#include "tests/dma/dma_test_common.h"

#ifndef EXPECTED_DMA_BASE
#error "Define EXPECTED_DMA_BASE to the target layout's DmaCore MMIO base"
#endif  // EXPECTED_DMA_BASE

#define TEST_SIZE 64  // 256 bytes: 16 full 16-byte beats
#define TAIL_BYTES 22u
#define TAIL_CANARY 0x5Au
#define ASYNC_SPIN_ITERS 256u

#define MCAUSE_LOAD_ACCESS_FAULT 5u
#define MCAUSE_STORE_ACCESS_FAULT 7u

static uint32_t dtcm_buf[TEST_SIZE] __attribute__((aligned(64)));

static volatile uint32_t *const ddr_src = (volatile uint32_t *)DDR_SRC;
static volatile uint32_t *const ddr_dst = (volatile uint32_t *)DDR_DST;
static volatile uint8_t *const ddr_src_bytes = (volatile uint8_t *)DDR_SRC;
static volatile uint8_t *const ddr_dst_bytes = (volatile uint8_t *)DDR_DST;

static volatile uint32_t g_trap_count;
static volatile uint32_t g_trap_mcause;
static volatile uint32_t g_trap_mtval;

#define REG32(addr) (*(volatile uint32_t *)(uintptr_t)(addr))

// Records synchronous access-fault CSRs and skips the 4-byte faulting insn.
__attribute__((aligned(4), naked)) static void dma_test_trap_handler(void) {
  __asm__ volatile(
      "csrr t0, mcause\n"
      "la   t1, g_trap_mcause\n"
      "sw   t0, 0(t1)\n"
      "csrr t0, mtval\n"
      "la   t1, g_trap_mtval\n"
      "sw   t0, 0(t1)\n"
      "la   t1, g_trap_count\n"
      "lw   t0, 0(t1)\n"
      "addi t0, t0, 1\n"
      "sw   t0, 0(t1)\n"
      "csrr t0, mepc\n"
      "addi t0, t0, 4\n"
      "csrw mepc, t0\n"
      "mret\n");
}

// Executes a 32-bit load from |addr| and verifies a LoadAccessFault trap.
static int expect_load_fault_u32(uint32_t addr) {
  g_trap_count = 0u;
  g_trap_mcause = 0u;
  g_trap_mtval = 0u;
  uint32_t val = 0x55AA55AAu;
  __asm__ volatile("lw %0, 0(%1)"
                   : "+r"(val)
                   : "r"(addr)
                   : "t0", "t1", "memory");
  return g_trap_count == 1u && g_trap_mcause == MCAUSE_LOAD_ACCESS_FAULT &&
         g_trap_mtval == addr && val == 0x55AA55AAu;
}

// Executes an 8-bit load from |addr| and verifies a LoadAccessFault trap.
static int expect_load_fault_u8(uint32_t addr) {
  g_trap_count = 0u;
  g_trap_mcause = 0u;
  g_trap_mtval = 0u;
  uint32_t val = 0x55AA55AAu;
  __asm__ volatile("lbu %0, 0(%1)"
                   : "+r"(val)
                   : "r"(addr)
                   : "t0", "t1", "memory");
  return g_trap_count == 1u && g_trap_mcause == MCAUSE_LOAD_ACCESS_FAULT &&
         g_trap_mtval == addr && val == 0x55AA55AAu;
}

// Executes a 32-bit store to |addr| and verifies a StoreAccessFault trap.
static int expect_store_fault_u32(uint32_t addr, uint32_t val) {
  g_trap_count = 0u;
  g_trap_mcause = 0u;
  g_trap_mtval = 0u;
  __asm__ volatile("sw %0, 0(%1)"
                   :
                   : "r"(val), "r"(addr)
                   : "t0", "t1", "memory");
  return g_trap_count == 1u && g_trap_mcause == MCAUSE_STORE_ACCESS_FAULT &&
         g_trap_mtval == addr;
}

// Executes an 8-bit store to |addr| and verifies a StoreAccessFault trap.
static int expect_store_fault_u8(uint32_t addr, uint32_t val) {
  g_trap_count = 0u;
  g_trap_mcause = 0u;
  g_trap_mtval = 0u;
  __asm__ volatile("sb %0, 0(%1)"
                   :
                   : "r"(val), "r"(addr)
                   : "t0", "t1", "memory");
  return g_trap_count == 1u && g_trap_mcause == MCAUSE_STORE_ACCESS_FAULT &&
         g_trap_mtval == addr;
}

// Executes a 2-register whole-register RVV load from |addr| and verifies a
// single LoadAccessFault trap at |addr| (proving beat-1 side effects halt).
static int expect_load_fault_vl2r(uint32_t addr) {
  g_trap_count = 0u;
  g_trap_mcause = 0u;
  g_trap_mtval = 0u;
  __asm__ volatile(
      ".option push\n"
      ".option arch, +zve32x\n"
      "vl2r.v v0, (%0)\n"
      ".option pop\n"
      :
      : "r"(addr)
      : "t0", "t1", "memory");
  return g_trap_count == 1u && g_trap_mcause == MCAUSE_LOAD_ACCESS_FAULT &&
         g_trap_mtval == addr;
}

// Executes a 2-register whole-register RVV store to |addr| and verifies a
// single StoreAccessFault trap at |addr| (proving beat-1 side effects halt).
static int expect_store_fault_vs2r(uint32_t addr) {
  g_trap_count = 0u;
  g_trap_mcause = 0u;
  g_trap_mtval = 0u;
  __asm__ volatile(
      ".option push\n"
      ".option arch, +zve32x\n"
      "vs2r.v v0, (%0)\n"
      ".option pop\n"
      :
      : "r"(addr)
      : "t0", "t1", "memory");
  return g_trap_count == 1u && g_trap_mcause == MCAUSE_STORE_ACCESS_FAULT &&
         g_trap_mtval == addr;
}

// Spins briefly so an erroneously started async 1-beat transfer would show up
// in STATUS or destination memory on asynchronous hardware as well as MPACT.
static void spin_for_async_check(void) {
  coralnpu_dma_fence();
  for (volatile uint32_t i = 0; i < ASYNC_SPIN_ITERS; ++i) {
  }
  coralnpu_dma_fence();
}

// Runs all DmaCore runtime checks; returns 0 on success or the failed step.
static int run(void) {
  // Verify __dma_base against the explicit build-supplied address and confirm
  // the aperture behaves as a live DmaCore CSR block rather than plain RAM
  // (XFER_CFG resets to 1, CTRL masks write-only CLEAR_ERROR on readback, and
  // coralnpu_dma_clear_error() preserves CTRL.ENABLE == 0 when disabled).
  if (coralnpu_dma_base() != EXPECTED_DMA_BASE) return 1;
  if (REG32(CORALNPU_DMA_CTRL) != 0u || REG32(CORALNPU_DMA_XFER_CFG) != 1u) {
    return 1;
  }
  coralnpu_dma_clear_error();
  if (REG32(CORALNPU_DMA_CTRL) != 0u) return 1;
  REG32(CORALNPU_DMA_CTRL) =
      CORALNPU_DMA_CTRL_ENABLE | CORALNPU_DMA_CTRL_CLEAR_ERROR;
  if (REG32(CORALNPU_DMA_CTRL) != CORALNPU_DMA_CTRL_ENABLE) return 1;
  REG32(CORALNPU_DMA_CTRL) = 0u;
  if (REG32(CORALNPU_DMA_CTRL) != 0u) return 1;

  for (int i = 0; i < TEST_SIZE; ++i) {
    ddr_src[i] = 0xA5A50000u + (uint32_t)i;
    dtcm_buf[i] = 0;
    ddr_dst[i] = 0;
  }

  // DDR -> DTCM.
  if (coralnpu_dma_copy_async((uint32_t)(uintptr_t)dtcm_buf, DDR_SRC,
                              sizeof(dtcm_buf)) != CORALNPU_DMA_OK) {
    return 2;
  }
  if (coralnpu_dma_wait_done() != CORALNPU_DMA_OK) return 3;
  for (int i = 0; i < TEST_SIZE; ++i) {
    if (dtcm_buf[i] != 0xA5A50000u + (uint32_t)i) return 4;
    dtcm_buf[i] += 1;
  }

  // DTCM -> DDR.
  if (coralnpu_dma_copy_async(DDR_DST, (uint32_t)(uintptr_t)dtcm_buf,
                              sizeof(dtcm_buf)) != CORALNPU_DMA_OK) {
    return 5;
  }
  if (coralnpu_dma_wait_done() != CORALNPU_DMA_OK) return 6;
  for (int i = 0; i < TEST_SIZE; ++i) {
    if (ddr_dst[i] != 0xA5A50001u + (uint32_t)i) return 7;
  }

  // Partial tail beat (22 bytes = 16 + 6): trailing bytes 22..31 must stay
  // untouched.
  for (uint32_t b = 0; b < 32u; ++b) {
    ddr_src_bytes[b] = (uint8_t)(0x80u + b);
    ddr_dst_bytes[b] = TAIL_CANARY;
  }
  if (coralnpu_dma_copy_async(DDR_DST, DDR_SRC, TAIL_BYTES) !=
      CORALNPU_DMA_OK) {
    return 8;
  }
  if (coralnpu_dma_wait_done() != CORALNPU_DMA_OK) return 9;
  for (uint32_t b = 0; b < TAIL_BYTES; ++b) {
    if (ddr_dst_bytes[b] != (uint8_t)(0x80u + b)) return 10;
  }
  for (uint32_t b = TAIL_BYTES; b < 32u; ++b) {
    if (ddr_dst_bytes[b] != TAIL_CANARY) return 11;
  }

  // A start issued while a transfer is outstanding must be refused.
  if (coralnpu_dma_copy_async((uint32_t)(uintptr_t)dtcm_buf, DDR_SRC,
                              sizeof(dtcm_buf)) != CORALNPU_DMA_OK) {
    return 12;
  }
  if (!coralnpu_dma_busy()) return 13;
  if (coralnpu_dma_copy_async((uint32_t)(uintptr_t)dtcm_buf, DDR_SRC,
                              sizeof(dtcm_buf)) != CORALNPU_DMA_EBUSY) {
    return 14;
  }
  if (coralnpu_dma_wait_done() != CORALNPU_DMA_OK) return 15;
  if (coralnpu_dma_busy()) return 16;

  // Length bounds: 0 and > 24-bit max (using +17 so 24-bit truncation would be
  // non-zero if unchecked).
  if (coralnpu_dma_copy_async(DDR_DST, DDR_SRC, 0u) != CORALNPU_DMA_EINVAL) {
    return 17;
  }
  if (coralnpu_dma_copy_async(DDR_DST, DDR_SRC, CORALNPU_DMA_LEN_MASK + 17u) !=
      CORALNPU_DMA_EINVAL) {
    return 18;
  }

  // Independent 16-byte alignment checks on dst and src (including 4B/8B
  // offsets that only fail the full 16B mask).
  if (coralnpu_dma_copy_async(DDR_DST + 4u, DDR_SRC, 64u) !=
      CORALNPU_DMA_EINVAL) {
    return 19;
  }
  if (coralnpu_dma_copy_async(DDR_DST, DDR_SRC + 8u, 64u) !=
      CORALNPU_DMA_EINVAL) {
    return 20;
  }

  // Unreachable tl_host targets (ITCM, Core CSR, DMA CSR, 32-bit address wrap,
  // and strided ITCM source) must report EHARDWARE with sticky XFER_ERROR.
  uint32_t buf_addr = (uint32_t)(uintptr_t)dtcm_buf;
  uint32_t csr_base = (EXPECTED_DMA_BASE == CORALNPU_DMA_HIGHMEM_BASE)
                          ? 0x00200000u
                          : 0x00030000u;
  uint32_t bad_addrs[4] = {
      0x00000000u,
      csr_base,
      EXPECTED_DMA_BASE,
      (EXPECTED_DMA_BASE == CORALNPU_DMA_HIGHMEM_BASE) ? 0x00002000u
                                                       : 0x00001FF0u,
  };
  for (uint32_t k = 0; k < 4u; ++k) {
    if (coralnpu_dma_copy_async(buf_addr, bad_addrs[k], 16u) !=
        CORALNPU_DMA_OK) {
      return 21;
    }
    if (coralnpu_dma_wait_done() != CORALNPU_DMA_EHARDWARE) return 22;
    if (coralnpu_dma_error_flags() != CORALNPU_DMA_STATUS_XFER_ERROR) return 23;
    coralnpu_dma_clear_error();
    if (coralnpu_dma_error_flags() != 0u ||
        REG32(CORALNPU_DMA_CTRL) != CORALNPU_DMA_CTRL_ENABLE) {
      return 24;
    }
  }

  if (coralnpu_dma_copy_async(0x00000000u, DDR_SRC, 64u) != CORALNPU_DMA_OK ||
      coralnpu_dma_wait_done() != CORALNPU_DMA_EHARDWARE ||
      coralnpu_dma_error_flags() != CORALNPU_DMA_STATUS_XFER_ERROR) {
    return 25;
  }
  coralnpu_dma_clear_error();

  // Two-beat transfer across 0xFFFFFFF0 -> 0x100000000 wraps 32-bit space.
  if (coralnpu_dma_copy_async(buf_addr, 0xFFFFFFF0u, 32u) != CORALNPU_DMA_OK ||
      coralnpu_dma_wait_done() != CORALNPU_DMA_EHARDWARE ||
      coralnpu_dma_error_flags() != CORALNPU_DMA_STATUS_XFER_ERROR) {
    return 26;
  }
  coralnpu_dma_clear_error();

  // Strided de-interleave xfer_error (both src and dst), plus sticky
  // XFER_ERROR blocking a valid START and single-store CTRL = ENABLE | START |
  // CLEAR_ERROR clearing the error and running a 16B copy.
  if (coralnpu_dma_deinterleave_async(buf_addr, 0x00000000u, /*size_bytes=*/32u,
                                      /*elem_width_log2=*/0u,
                                      /*n_areas=*/2u) != CORALNPU_DMA_OK ||
      coralnpu_dma_wait_done() != CORALNPU_DMA_EHARDWARE ||
      coralnpu_dma_error_flags() != CORALNPU_DMA_STATUS_XFER_ERROR) {
    return 27;
  }
  coralnpu_dma_clear_error();
  if (coralnpu_dma_deinterleave_async(0x00000000u, DDR_SRC, /*size_bytes=*/32u,
                                      /*elem_width_log2=*/0u,
                                      /*n_areas=*/2u) != CORALNPU_DMA_OK ||
      coralnpu_dma_wait_done() != CORALNPU_DMA_EHARDWARE ||
      coralnpu_dma_error_flags() != CORALNPU_DMA_STATUS_XFER_ERROR) {
    return 28;
  }
  ddr_src[0] = 0x55667788u;
  ddr_dst[0] = 0x99AABBCCu;
  REG32(CORALNPU_DMA_SRC_ADDR) = DDR_SRC;
  REG32(CORALNPU_DMA_DST_ADDR) = DDR_DST;
  REG32(CORALNPU_DMA_LEN_FLAGS) = 16u;
  REG32(CORALNPU_DMA_XFER_CFG) = 1u;
  REG32(CORALNPU_DMA_CTRL) = CORALNPU_DMA_CTRL_ENABLE | CORALNPU_DMA_CTRL_START;
  if ((REG32(CORALNPU_DMA_STATUS) &
       (CORALNPU_DMA_STATUS_BUSY | CORALNPU_DMA_STATUS_DONE)) != 0u) {
    return 28;
  }
  spin_for_async_check();
  if (coralnpu_dma_error_flags() != CORALNPU_DMA_STATUS_XFER_ERROR ||
      ddr_dst[0] != 0x99AABBCCu) {
    return 28;
  }
  REG32(CORALNPU_DMA_CTRL) = CORALNPU_DMA_CTRL_ENABLE |
                             CORALNPU_DMA_CTRL_START |
                             CORALNPU_DMA_CTRL_CLEAR_ERROR;
  while ((REG32(CORALNPU_DMA_STATUS) & CORALNPU_DMA_STATUS_BUSY) != 0u ||
         (REG32(CORALNPU_DMA_STATUS) &
          (CORALNPU_DMA_STATUS_DONE | CORALNPU_DMA_STATUS_ERROR)) == 0u) {
  }
  coralnpu_dma_fence();
  if (REG32(CORALNPU_DMA_STATUS) != CORALNPU_DMA_STATUS_DONE ||
      ddr_dst[0] != 0x55667788u) {
    return 28;
  }

  // Direct MMIO checks: START with ENABLE=0 is ignored (even after later
  // asserting ENABLE alone), independent SRC/DST align_error, sticky error_eff
  // blocking a valid START without CLEAR_ERROR, and each cfg_error clause.
  ddr_src[0] = 0x778899AAu;
  ddr_dst[0] = 0x11223344u;
  REG32(CORALNPU_DMA_SRC_ADDR) = DDR_SRC;
  REG32(CORALNPU_DMA_DST_ADDR) = DDR_DST;
  REG32(CORALNPU_DMA_LEN_FLAGS) = 16u;
  REG32(CORALNPU_DMA_XFER_CFG) = 1u;
  REG32(CORALNPU_DMA_CTRL) = CORALNPU_DMA_CTRL_START;
  if ((REG32(CORALNPU_DMA_STATUS) & CORALNPU_DMA_STATUS_BUSY) != 0u) return 29;
  spin_for_async_check();
  REG32(CORALNPU_DMA_CTRL) = CORALNPU_DMA_CTRL_ENABLE;
  spin_for_async_check();
  if (REG32(CORALNPU_DMA_STATUS) != CORALNPU_DMA_STATUS_DONE ||
      ddr_dst[0] != 0x11223344u) {
    return 29;
  }

  REG32(CORALNPU_DMA_SRC_ADDR) = DDR_SRC + 4u;
  REG32(CORALNPU_DMA_DST_ADDR) = DDR_DST;
  REG32(CORALNPU_DMA_CTRL) = CORALNPU_DMA_CTRL_ENABLE | CORALNPU_DMA_CTRL_START;
  if (coralnpu_dma_error_flags() != CORALNPU_DMA_STATUS_ALIGN_ERROR) return 30;
  coralnpu_dma_clear_error();

  REG32(CORALNPU_DMA_SRC_ADDR) = DDR_SRC;
  REG32(CORALNPU_DMA_DST_ADDR) = DDR_DST + 4u;
  REG32(CORALNPU_DMA_CTRL) = CORALNPU_DMA_CTRL_ENABLE | CORALNPU_DMA_CTRL_START;
  if (coralnpu_dma_error_flags() != CORALNPU_DMA_STATUS_ALIGN_ERROR) return 31;

  // Without CLEAR_ERROR, a valid START must be ignored, keep BUSY/DONE low on
  // both synchronous and asynchronous hardware, and retain ALIGN_ERROR.
  ddr_dst[0] = 0x11223344u;
  REG32(CORALNPU_DMA_DST_ADDR) = DDR_DST;
  REG32(CORALNPU_DMA_CTRL) = CORALNPU_DMA_CTRL_ENABLE | CORALNPU_DMA_CTRL_START;
  if ((REG32(CORALNPU_DMA_STATUS) &
       (CORALNPU_DMA_STATUS_BUSY | CORALNPU_DMA_STATUS_DONE)) != 0u) {
    return 32;
  }
  spin_for_async_check();
  if ((REG32(CORALNPU_DMA_STATUS) &
       (CORALNPU_DMA_STATUS_BUSY | CORALNPU_DMA_STATUS_DONE)) != 0u ||
      ddr_dst[0] != 0x11223344u ||
      coralnpu_dma_error_flags() != CORALNPU_DMA_STATUS_ALIGN_ERROR) {
    return 32;
  }
  coralnpu_dma_clear_error();

  // Independent cfg_error clauses: bad_mode, bad_elem_size, len==0,
  // (strided && area_size==0 via 24-bit AREA_SIZE mask), n_areas==0, and
  // n_areas==17, plus CSR readbacks on DST_ADDR, LEN_FLAGS, and AREA_SIZE.
  REG32(CORALNPU_DMA_AREA_SIZE) = 0x01ABCDEFu;
  if (REG32(CORALNPU_DMA_AREA_SIZE) != 0x00ABCDEFu ||
      REG32(CORALNPU_DMA_DST_ADDR) != DDR_DST) {
    return 33;
  }
  const struct {
    uint32_t len_flags;
    uint32_t n_areas;
    uint32_t area_size;
  } bad_cfgs[6] = {
      {16u | (1u << 27), 1u, 0u},
      {16u | (5u << 24), 1u, 0u},
      {0u, 1u, 0u},
      {32u, 2u, 0x01000000u},
      {16u, 0u, 0u},
      {32u, 17u, 16u},
  };
  for (uint32_t k = 0; k < 6u; ++k) {
    REG32(CORALNPU_DMA_LEN_FLAGS) = bad_cfgs[k].len_flags;
    REG32(CORALNPU_DMA_XFER_CFG) = bad_cfgs[k].n_areas;
    REG32(CORALNPU_DMA_AREA_SIZE) = bad_cfgs[k].area_size;
    if (REG32(CORALNPU_DMA_LEN_FLAGS) != bad_cfgs[k].len_flags ||
        REG32(CORALNPU_DMA_AREA_SIZE) !=
            (bad_cfgs[k].area_size & CORALNPU_DMA_LEN_MASK)) {
      return 33;
    }
    REG32(CORALNPU_DMA_CTRL) =
        CORALNPU_DMA_CTRL_ENABLE | CORALNPU_DMA_CTRL_START;
    if (coralnpu_dma_error_flags() != CORALNPU_DMA_STATUS_CFG_ERROR) return 33;
    if (k + 1u < 6u) coralnpu_dma_clear_error();
  }
  REG32(CORALNPU_DMA_AREA_SIZE) = 0u;

  // Leaving CFG_ERROR latched from k=5: next driver transfer must NOT silently
  // clear it (RTL Requirement 8); wait_done() must report EHARDWARE with
  // CFG_ERROR preserved and destination untouched until explicit clear_error().
  ddr_dst[0] = 0xCAFEBABEu;
  if (coralnpu_dma_copy_async(DDR_DST, DDR_SRC, 64u) != CORALNPU_DMA_OK) {
    return 34;
  }
  if (coralnpu_dma_wait_done() != CORALNPU_DMA_EHARDWARE ||
      coralnpu_dma_error_flags() != CORALNPU_DMA_STATUS_CFG_ERROR ||
      ddr_dst[0] != 0xCAFEBABEu) {
    return 35;
  }
  coralnpu_dma_clear_error();
  if (coralnpu_dma_copy_async(DDR_DST, DDR_SRC, 64u) != CORALNPU_DMA_OK ||
      coralnpu_dma_wait_done() != CORALNPU_DMA_OK ||
      coralnpu_dma_error_flags() != 0u || ddr_dst[0] != ddr_src[0]) {
    return 36;
  }

  // DmaCsr.scala sets d_error_reg := !addrValid on CSR offsets outside
  // {0x00..0x1c} (e.g. >= 0x20 or byte offsets +0x01/+0x09), raising load/store
  // access faults, while 32-bit writes to RO registers (STATUS, PASS_REM) are
  // ignored without faulting. Under DEFERRED (MPACT), sub-word and misaligned
  // 32-bit accesses inside 0x00..0x1c also fault per README §2.3.
  uint32_t saved_mtvec = 0u;
  __asm__ volatile("csrrw %0, mtvec, %1"
                   : "=r"(saved_mtvec)
                   : "r"((uintptr_t)dma_test_trap_handler)
                   : "memory");
  uint32_t saved_ctrl = REG32(CORALNPU_DMA_CTRL);
  uint32_t saved_src = REG32(CORALNPU_DMA_SRC_ADDR);
  int fault_ok =
      expect_load_fault_u32(EXPECTED_DMA_BASE + 0x20u) &&
      expect_load_fault_u32(EXPECTED_DMA_BASE + 0xFFCu) &&
      expect_load_fault_u8(EXPECTED_DMA_BASE + 0x01u) &&
      expect_store_fault_u32(EXPECTED_DMA_BASE + 0x20u, 0x12345678u) &&
      expect_store_fault_u8(CORALNPU_DMA_SRC_ADDR + 0x01u, 0xFFu) &&
      expect_load_fault_vl2r(EXPECTED_DMA_BASE + 0x20u) &&
      expect_store_fault_vs2r(EXPECTED_DMA_BASE + 0x20u);
#if CORALNPU_DMA_ISSUE_MODE == CORALNPU_DMA_ISSUE_DEFERRED
  fault_ok = fault_ok && expect_load_fault_u32(EXPECTED_DMA_BASE + 0x02u) &&
             expect_load_fault_u8(CORALNPU_DMA_STATUS) &&
             expect_store_fault_u32(EXPECTED_DMA_BASE + 0x02u, 0x12345678u) &&
             expect_store_fault_u8(CORALNPU_DMA_SRC_ADDR, 0xFFu);
#endif  // CORALNPU_DMA_ISSUE_MODE
  fault_ok = fault_ok && REG32(CORALNPU_DMA_CTRL) == saved_ctrl &&
             REG32(CORALNPU_DMA_SRC_ADDR) == saved_src;
  g_trap_count = 0u;
  REG32(CORALNPU_DMA_STATUS) = 0xFFFFFFFFu;
  REG32(CORALNPU_DMA_PASS_REM) = 0xFFFFFFFFu;
  fault_ok = fault_ok && (g_trap_count == 0u) &&
             REG32(CORALNPU_DMA_STATUS) == CORALNPU_DMA_STATUS_DONE &&
             REG32(CORALNPU_DMA_PASS_REM) == 0u;
  if (EXPECTED_DMA_BASE == CORALNPU_DMA_HIGHMEM_BASE) {
    // On highmem builds 0x00032000 is ITCM, not DmaCore: a 32-bit read at
    // 0x00032020 must not trap as an undefined DmaCore offset.
    volatile uint32_t itcm_word = REG32(CORALNPU_DMA_DEFAULT_BASE + 0x20u);
    (void)itcm_word;
    fault_ok = fault_ok && (g_trap_count == 0u);
  }
  __asm__ volatile("csrw mtvec, %0" : : "r"(saved_mtvec) : "memory");
  if (!fault_ok) return 37;

  // A wait with nothing outstanding is a no-op, not an error.
  if (coralnpu_dma_wait_done() != CORALNPU_DMA_OK) return 38;

  return 0;
}

int main(void) {
  *dma_test_result = RESULT_TAG_RUNTIME | (uint32_t)run();
  return 0;
}
