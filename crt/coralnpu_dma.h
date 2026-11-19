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

// Software contract for the CoralNPU core-local DMA engine (dma.DmaCore).
//
// The register map and transfer geometry below match the RTL specification
// (third_party/coralnpu/hdl/chisel/src/dma/README.md) and implementation
// (DmaCore.scala, DmaCsr.scala, CoreAxi.scala, Parameters.scala).
// Hardware properties that callers (and compiler codegen) must respect:
//
//   1. SRC_ADDR and DST_ADDR must be 16-byte aligned (align_error otherwise).
//      Byte length need not be a multiple of 16; hardware masks the tail beat.
//   2. 0 < len <= 0x00FFFFFF. In strided de-interleave mode (n_areas = 2..16),
//      len must be a multiple of (n_areas << elem_width_log2) and AREA_SIZE is
//      set to len / n_areas.
//   3. elem_width_log2 <= 4, 1 <= n_areas <= 16, and LEN_FLAGS[31:27] == 0
//      (cfg_error otherwise, except reserved LEN_FLAGS[31:30]).
//   4. In strided mode, source must remain readable up to ceil(area_size / 16)
//      * n_areas * 16 bytes (over-read up to 240 bytes; contents ignored).
//   5. In strided mode, destination must hold n_areas * row_pitch bytes, where
//      row_pitch is ceil(area_size / 16) * 16.
//   6. DmaCore's host port reaches banked DTCM and external AXI (EXTMEM/DDR);
//      ITCM (including .rodata) and core/DMA CSRs return xfer_error.
//   7. STATUS error bits are sticky until cleared by CTRL.CLEAR_ERROR (blocking
//      subsequent starts), STATUS.done rises only on clean completion, and each
//      start must be paired with one wait_done() (single transfer in flight).

#ifndef CRT_CORALNPU_DMA_H_
#define CRT_CORALNPU_DMA_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// MMIO base addresses (Parameters.scala: csrRegion.memStart + 0x2000).
#define CORALNPU_DMA_DEFAULT_BASE 0x00032000u
#define CORALNPU_DMA_HIGHMEM_BASE 0x00202000u

// Returns the DmaCore MMIO base address for the linked memory layout
// (0x00032000 on default 32 KB DTCM, 0x00202000 on 1 MB highmem).
#if defined(__riscv)
extern char __dma_base[];
static inline uint32_t coralnpu_dma_base(void) {
  return (uint32_t)(uintptr_t)__dma_base;
}
#else
static inline uint32_t coralnpu_dma_base(void) {
  return CORALNPU_DMA_DEFAULT_BASE;
}
#endif  // __riscv

// Register offsets from coralnpu_dma_base() (DmaCsr.scala).
#define CORALNPU_DMA_CTRL_OFFSET 0x00u
#define CORALNPU_DMA_STATUS_OFFSET 0x04u
#define CORALNPU_DMA_SRC_ADDR_OFFSET 0x08u
#define CORALNPU_DMA_DST_ADDR_OFFSET 0x0Cu
#define CORALNPU_DMA_LEN_FLAGS_OFFSET 0x10u
#define CORALNPU_DMA_XFER_CFG_OFFSET 0x14u
#define CORALNPU_DMA_AREA_SIZE_OFFSET 0x18u
#define CORALNPU_DMA_PASS_REM_OFFSET 0x1Cu

// Absolute MMIO register addresses for the active memory layout.
#define CORALNPU_DMA_CTRL (coralnpu_dma_base() + CORALNPU_DMA_CTRL_OFFSET)
#define CORALNPU_DMA_STATUS (coralnpu_dma_base() + CORALNPU_DMA_STATUS_OFFSET)
#define CORALNPU_DMA_SRC_ADDR \
  (coralnpu_dma_base() + CORALNPU_DMA_SRC_ADDR_OFFSET)
#define CORALNPU_DMA_DST_ADDR \
  (coralnpu_dma_base() + CORALNPU_DMA_DST_ADDR_OFFSET)
#define CORALNPU_DMA_LEN_FLAGS \
  (coralnpu_dma_base() + CORALNPU_DMA_LEN_FLAGS_OFFSET)
#define CORALNPU_DMA_XFER_CFG \
  (coralnpu_dma_base() + CORALNPU_DMA_XFER_CFG_OFFSET)
#define CORALNPU_DMA_AREA_SIZE \
  (coralnpu_dma_base() + CORALNPU_DMA_AREA_SIZE_OFFSET)
#define CORALNPU_DMA_PASS_REM \
  (coralnpu_dma_base() + CORALNPU_DMA_PASS_REM_OFFSET)

// CTRL register bits (reads return ENABLE only; CTRL[2] abort is reserved).
#define CORALNPU_DMA_CTRL_ENABLE (1u << 0)
#define CORALNPU_DMA_CTRL_START (1u << 1)        // self-clearing from idle
#define CORALNPU_DMA_CTRL_CLEAR_ERROR (1u << 3)  // clears sticky error flags

// STATUS register bits.
#define CORALNPU_DMA_STATUS_BUSY (1u << 0)
#define CORALNPU_DMA_STATUS_DONE (1u << 1)
#define CORALNPU_DMA_STATUS_ERROR (1u << 2)
#define CORALNPU_DMA_STATUS_ALIGN_ERROR (1u << 3)
#define CORALNPU_DMA_STATUS_CFG_ERROR (1u << 4)
#define CORALNPU_DMA_STATUS_XFER_ERROR (1u << 5)
#define CORALNPU_DMA_STATUS_ERROR_FLAGS_MASK                         \
  (CORALNPU_DMA_STATUS_ALIGN_ERROR | CORALNPU_DMA_STATUS_CFG_ERROR | \
   CORALNPU_DMA_STATUS_XFER_ERROR)

// Bus beat geometry and LEN_FLAGS / XFER_CFG limits (DLENB = 16 bytes).
#define CORALNPU_DMA_BEAT_BYTES 16u
#define CORALNPU_DMA_ALIGN_MASK (CORALNPU_DMA_BEAT_BYTES - 1u)
#define CORALNPU_DMA_LEN_MASK 0x00FFFFFFu
#define CORALNPU_DMA_WIDTH_SHIFT 24u
#define CORALNPU_DMA_WIDTH_MASK 0x7u
#define CORALNPU_DMA_MAX_WIDTH_LOG2 4u
#define CORALNPU_DMA_MAX_AREAS 16u

// Return codes for the CRT helpers.
#define CORALNPU_DMA_OK 0
#define CORALNPU_DMA_EBUSY (-1)
#define CORALNPU_DMA_EINVAL (-2)
#define CORALNPU_DMA_ETIMEOUT (-3)
#define CORALNPU_DMA_EHARDWARE (-4)

// Packs LEN_FLAGS ([23:0] xfer_len, [26:24] xfer_width; [31:27] must be 0).
static inline uint32_t coralnpu_dma_make_len_flags(uint32_t len,
                                                   uint32_t elem_width_log2) {
  return (len & CORALNPU_DMA_LEN_MASK) |
         ((elem_width_log2 & CORALNPU_DMA_WIDTH_MASK)
          << CORALNPU_DMA_WIDTH_SHIFT);
}

// Returns the byte stride between output planes in strided de-interleave mode:
// ceil(area_size / 16) * 16.
static inline uint32_t coralnpu_dma_row_pitch(uint32_t area_size) {
  return (area_size + CORALNPU_DMA_ALIGN_MASK) & ~CORALNPU_DMA_ALIGN_MASK;
}

// Compiler + hardware barrier: orders normal memory stores before the MMIO
// CTRL.START store (w -> o), and orders the MMIO STATUS load before subsequent
// reads of the destination buffer (i -> r).
static inline void coralnpu_dma_fence(void) {
#if defined(__riscv)
  __asm__ volatile("fence iorw, iorw" ::: "memory");
#else
  __asm__ volatile("" ::: "memory");
#endif  // __riscv
}

// Issue mode. Deferred issue models transfer latency when the functional
// simulator completes transfers synchronously inside CTRL.START: start calls
// only record the job and wait_done() issues it after a modelled delay so
// premature reads of the destination remain observable. Immediate issue writes
// the MMIO registers on start.
#define CORALNPU_DMA_ISSUE_IMMEDIATE 0
#define CORALNPU_DMA_ISSUE_DEFERRED 1
#ifndef CORALNPU_DMA_ISSUE_MODE
#define CORALNPU_DMA_ISSUE_MODE CORALNPU_DMA_ISSUE_DEFERRED
#endif  // CORALNPU_DMA_ISSUE_MODE

// Event trace, written by the guest and read back by the test runner after the
// run. Word 0 of the region is the record count; records follow.
typedef struct coralnpu_dma_trace_t {
  uint32_t seq;    // transfer sequence number
  uint32_t phase;  // CORALNPU_DMA_EVENT_*
  uint32_t bytes;  // total transfer payload bytes
  uint32_t stamp;  // cycle counter, or 0 when unavailable
} coralnpu_dma_trace_t;

// Trace record is only part of this backend for now.
// We need this because RISC-V program doesn't have stdout or OS I/O.
// This trace buffer acts as a mailbox to the host to verify that DMA
// operations are executed correctly. We might not need these at all
// in the future, once the higher level compiler codegen is implemented.
// Until then, we are keeping traces. Traces are recorded in ddr_mem.
// starting address: 0x80003000, first record at: 0x80003004.
#if defined(__cplusplus)
static_assert(sizeof(coralnpu_dma_trace_t) == 16,
              "DMA trace record must be exactly 16 bytes");
#else
_Static_assert(sizeof(coralnpu_dma_trace_t) == 16,
               "DMA trace record must be exactly 16 bytes");
#endif

#define CORALNPU_DMA_EVENT_START 1u
#define CORALNPU_DMA_EVENT_DONE 2u

// Kept in DDR rather than DTCM: 4,100 bytes in DTCM would shrink the 32 KB
// DTCM firmware heap below the 6-page (24 KB) workgroup local-memory capacity
// tested by local_memory_size_test. runtime/sim/simulator_inline.c starts HAL
// bindings above this region and zeroes the count per dispatch. Both
// libcoralnpu_crt.a and simulator_inline.c compile against these defaults, so
// do not override them per translation unit.
// TODO: Remove the guest trace (or default CORALNPU_DMA_TRACE_CAPACITY to 0)
// once DMA codegen is in place.
#ifndef CORALNPU_DMA_TRACE_ADDR
#define CORALNPU_DMA_TRACE_ADDR 0x80003000u
#endif  // CORALNPU_DMA_TRACE_ADDR
#ifndef CORALNPU_DMA_TRACE_CAPACITY
#define CORALNPU_DMA_TRACE_CAPACITY 256u
#endif  // CORALNPU_DMA_TRACE_CAPACITY

// Reads the cycle counter when the guest has one. The CRT runs in M-mode and
// already reads minstret via csrr, so mcycle is used rather than the U-mode
// rdcycle alias. Off by default until confirmed against the MPACT CSR set.
static inline uint32_t coralnpu_dma_stamp(void) {
#if defined(__riscv) && defined(CORALNPU_DMA_HAVE_RDCYCLE)
  uint32_t c;
  __asm__ volatile("csrr %0, mcycle" : "=r"(c));
  return c;
#else
  return 0u;
#endif  // __riscv && CORALNPU_DMA_HAVE_RDCYCLE
}

// Returns nonzero while a transfer is outstanding (awaiting wait_done() or
// still active in STATUS.busy).
int coralnpu_dma_busy(void);

// Reads STATUS and returns the sticky error flag bits
// (CORALNPU_DMA_STATUS_{ALIGN,CFG,XFER}_ERROR; 0 when no error).
uint32_t coralnpu_dma_error_flags(void);

// Clears sticky STATUS error flags via CTRL.CLEAR_ERROR while preserving
// CTRL.ENABLE.
void coralnpu_dma_clear_error(void);

// Starts an asynchronous transfer of |size_bytes| from |src| to |dst|,
// optionally de-interleaving elements of size (1 << elem_width_log2) bytes
// across |n_areas| output planes (1 = contiguous memcopy, 2..16 = strided
// de-interleave with plane pitch coralnpu_dma_row_pitch(size_bytes / n_areas)).
// Returns CORALNPU_DMA_EINVAL on invalid alignment/shape, CORALNPU_DMA_EBUSY
// if a previous transfer has not been awaited, or CORALNPU_DMA_OK.
int coralnpu_dma_deinterleave_async(uint32_t dst, uint32_t src,
                                    uint32_t size_bytes,
                                    uint32_t elem_width_log2, uint32_t n_areas);

// Starts an asynchronous contiguous 1D memory copy of |size_bytes| from |src|
// to |dst|. Both addresses must be 16-byte aligned; |size_bytes| may be any
// value in 1..CORALNPU_DMA_LEN_MASK.
int coralnpu_dma_copy_async(uint32_t dst, uint32_t src, uint32_t size_bytes);

// Waits for the outstanding transfer to finish. Returns CORALNPU_DMA_OK (also
// when idle), CORALNPU_DMA_EHARDWARE if STATUS.error is set (inspect flags via
// coralnpu_dma_error_flags() and clear via coralnpu_dma_clear_error() before
// starting another transfer), or CORALNPU_DMA_ETIMEOUT.
int coralnpu_dma_wait_done(void);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // CRT_CORALNPU_DMA_H_
