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

// Software contract for the CoralNPU DMA engine.
//
// The register map and descriptor layout below are taken from the hardware
// specification (third_party/coralnpu/doc/peripherals/dma.md) and its
// implementation (third_party/coralnpu/hdl/chisel/src/bus/DmaEngine.scala).
// Hardware properties that callers (and later, compiler codegen) must respect:
//
//   1. Descriptors are 32 bytes and 32-byte aligned; the engine fetches them
//      as two 128-bit beats.
//   2. xfer_len must be an exact multiple of the beat size and both addresses
//      must be beat-aligned, otherwise the engine's 24-bit remaining counter
//      underflows and it transfers ~16 MB.
//   3. A descriptor chain must not be modified while it is in flight.
//   4. There is one channel and one outstanding transaction; concurrent
//      transfers must be issued as one chain or serialised by the caller.
//   5. STATUS.done is level-triggered: it stays set until the next start from
//      idle, so always pair a start with exactly one wait.
//   6. poll_en requires poll_addr != 0.

#ifndef CRT_CORALNPU_DMA_H_
#define CRT_CORALNPU_DMA_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// MMIO base address and register offsets.
#define CORALNPU_DMA_BASE 0x40050000u
#define CORALNPU_DMA_CTRL (CORALNPU_DMA_BASE + 0x00u)
#define CORALNPU_DMA_STATUS (CORALNPU_DMA_BASE + 0x04u)
#define CORALNPU_DMA_DESC_ADDR (CORALNPU_DMA_BASE + 0x08u)
#define CORALNPU_DMA_CUR_DESC (CORALNPU_DMA_BASE + 0x0Cu)
#define CORALNPU_DMA_XFER_REMAIN (CORALNPU_DMA_BASE + 0x10u)

// CTRL register bits.
#define CORALNPU_DMA_CTRL_ENABLE (1u << 0)
#define CORALNPU_DMA_CTRL_START (1u << 1)  // W1S, self-clearing
#define CORALNPU_DMA_CTRL_ABORT (1u << 2)

// STATUS register bits.
#define CORALNPU_DMA_STATUS_BUSY (1u << 0)
#define CORALNPU_DMA_STATUS_DONE (1u << 1)
#define CORALNPU_DMA_STATUS_ERROR (1u << 2)
// STATUS[7:4] holds the error code; shift then mask to extract it.
#define CORALNPU_DMA_STATUS_ERROR_CODE_SHIFT 4u
#define CORALNPU_DMA_STATUS_ERROR_CODE_MASK 0xFu

// Error codes reported in STATUS[7:4] (DmaEngine.scala status update).
#define CORALNPU_DMA_ERR_NONE 0u
#define CORALNPU_DMA_ERR_DESC_FETCH 1u  // sFetchDesc0Resp / sFetchDesc1Resp
#define CORALNPU_DMA_ERR_POLL 2u        // sPollResp
#define CORALNPU_DMA_ERR_READ 3u        // sXferReadResp
#define CORALNPU_DMA_ERR_WRITE 4u       // sXferWriteResp
#define CORALNPU_DMA_ERR_ABORT 5u       // CTRL.abort

// Descriptor flag bits within the packed len_flags word.
#define CORALNPU_DMA_LEN_MASK 0x00FFFFFFu
#define CORALNPU_DMA_WIDTH_SHIFT 24u
#define CORALNPU_DMA_WIDTH_MASK 0x7u
#define CORALNPU_DMA_FLAG_SRC_FIXED (1u << 27)
#define CORALNPU_DMA_FLAG_DST_FIXED (1u << 28)
#define CORALNPU_DMA_FLAG_POLL_EN (1u << 29)
#define CORALNPU_DMA_FLAGS_MASK                                    \
  (CORALNPU_DMA_FLAG_SRC_FIXED | CORALNPU_DMA_FLAG_DST_FIXED | \
   CORALNPU_DMA_FLAG_POLL_EN)

// Beat size is log2(bytes); the 128-bit host port caps it at 4 (16 bytes).
#define CORALNPU_DMA_MAX_WIDTH_LOG2 4u
// Narrowest beat the contract accepts, as log2(bytes). Each beat costs a bus
// read/write round trip, so a 1-byte beat is ~16x slower than a 16-byte one for
// the same payload. Compiler-generated tiles are element-aligned (4 bytes or
// wider), so needing a narrower beat means the caller computed a bad address or
// length -- reject it rather than silently degrade.
#define CORALNPU_DMA_MIN_WIDTH_LOG2 2u
// Returned by coralnpu_dma_choose_width() when no accepted beat size fits.
#define CORALNPU_DMA_WIDTH_INVALID 0xFFFFFFFFu

// Return codes for the CRT helpers (distinct from hardware error codes).
#define CORALNPU_DMA_OK 0
#define CORALNPU_DMA_EBUSY (-1)
#define CORALNPU_DMA_EINVAL (-2)
#define CORALNPU_DMA_ETIMEOUT (-3)
#define CORALNPU_DMA_EHARDWARE (-4)

// 32-byte DMA descriptor; must be 32-byte aligned in memory.
typedef struct __attribute__((packed, aligned(32))) coralnpu_dma_descriptor_t {
  uint32_t src_addr;
  uint32_t dst_addr;
  uint32_t len_flags;  // [23:0] len, [26:24] log2(beat bytes), [27] src_fixed,
                       // [28] dst_fixed, [29] poll_en
  uint32_t next_desc;  // address of next descriptor, or 0 for end of chain
  uint32_t poll_addr;  // status register polled before every beat (0 = none)
  uint32_t poll_mask;
  uint32_t poll_value;
  uint32_t reserved;
} coralnpu_dma_descriptor_t;


// The length of the DMA descriptor needs to be 32 bytes:
// Ref: third_party/coralnpu/doc/peripherals/dma.md
#if defined(__cplusplus)
static_assert(sizeof(coralnpu_dma_descriptor_t) == 32,
              "DMA descriptor must be exactly 32 bytes");
#else
_Static_assert(sizeof(coralnpu_dma_descriptor_t) == 32,
               "DMA descriptor must be exactly 32 bytes");
#endif

// Constructs the packed len_flags field. |flags| is a bitwise OR of the
// CORALNPU_DMA_FLAG_* values.
static inline uint32_t coralnpu_dma_make_len_flags(uint32_t len,
                                                   uint32_t width_log2,
                                                   uint32_t flags) {
  return (len & CORALNPU_DMA_LEN_MASK) |
         ((width_log2 & CORALNPU_DMA_WIDTH_MASK) << CORALNPU_DMA_WIDTH_SHIFT) |
         (flags & CORALNPU_DMA_FLAGS_MASK);
}

// Largest accepted beat size (as log2 bytes) for a transfer, or
// CORALNPU_DMA_WIDTH_INVALID if none fits. The engine decrements its
// remaining-length counter by exactly one beat per iteration and stops only
// when the counter reaches zero, so the beat size MUST divide the length;
// addresses must be beat-aligned for the same reason. Searching stops at
// CORALNPU_DMA_MIN_WIDTH_LOG2 rather than falling through to a 1-byte beat, so
// a badly shaped transfer fails loudly instead of running 16x slow.
// TODO: should we fail or continue with a slow transfer?
static inline uint32_t coralnpu_dma_choose_width(uint32_t len, uint32_t src,
                                                 uint32_t dst) {
  uint32_t w = CORALNPU_DMA_MAX_WIDTH_LOG2 + 1u;
  while (w-- > CORALNPU_DMA_MIN_WIDTH_LOG2) {
    uint32_t beat = 1u << w;
    if ((len % beat) == 0u && (src % beat) == 0u && (dst % beat) == 0u) {
      return w;
    }
  }
  return CORALNPU_DMA_WIDTH_INVALID;
}

// Compiler + hardware barrier, used at two points that need different
// orderings: before CTRL.start, the descriptor's ordinary memory stores must be
// visible before the device store (w -> o); after STATUS.done is observed, the
// device load must precede the caller's loads of the destination (i -> r). A
// full fence covers both and is accepted by every RISC-V assembler.
static inline void coralnpu_dma_fence(void) {
#if defined(__riscv)
  __asm__ volatile("fence iorw, iorw" ::: "memory");
#else
  __asm__ volatile("" ::: "memory");
#endif  // __riscv
}

// Issue mode. Deferred issue models transfer latency without simulator
// support: start() only records the chain and wait_done() issues it after a
// modelled delay. Immediate issue is the hardware behaviour. The API and the
// generated call sequence are identical in both modes.
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
  uint32_t bytes;  // total chain payload bytes
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

// True while a chain is outstanding. Consults software state under deferred
// issue and STATUS.busy under immediate issue.
int coralnpu_dma_busy(void);

// Reads STATUS and extracts the hardware error code (0 when no error).
uint32_t coralnpu_dma_error_code(void);

// Kicks off a descriptor chain. Returns CORALNPU_DMA_OK, or CORALNPU_DMA_EBUSY,
// if a previous chain is still in flight (or CORALNPU_DMA_EINVAL if the chain
// walk fails). The engine latches a start written while busy and re-fires it
// on its next return to idle, so skipping this check would silently run a second
// transfer of whatever DESC_ADDR holds then.
int coralnpu_dma_start(const coralnpu_dma_descriptor_t *desc);

// Polling barrier. Returns CORALNPU_DMA_OK on clean completion,
// CORALNPU_DMA_EHARDWARE if STATUS.error is set (query the code with
// coralnpu_dma_error_code()), or CORALNPU_DMA_ETIMEOUT.
int coralnpu_dma_wait_done(void);

// Aborts an in-flight chain and returns the engine to idle.
void coralnpu_dma_abort(void);

// Single non-blocking mem-to-mem transfer. Selects the beat size, validates
// it, fills |desc|, and starts the engine. Returns CORALNPU_DMA_EINVAL if the
// transfer cannot be expressed, otherwise the result of coralnpu_dma_start().
int coralnpu_dma_copy_async(uint32_t dst, uint32_t src, uint32_t size_bytes,
                            coralnpu_dma_descriptor_t *desc);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // CRT_CORALNPU_DMA_H_
