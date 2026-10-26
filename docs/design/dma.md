# CoralNPU DMA: Architecture, Contract, and Driver

## Overview

CoralNPU executes compute kernels out of fast, tightly coupled on-core memory
(**DTCM**, 32 KB, 1-cycle latency at 16 B/cycle), while model inputs and
outputs live in off-chip **DDR** (~75–160 ns per beat). When generated code
stages tiles in DTCM (§7.2), it copies them mostly with synchronous RVV loops
that stall the core on memory latency. Currently, the generated code does not
call the DMA driver; only `tests/dma` calls it.

This doc covers the **DMA engine** (`DmaEngine`) and its CRT driver. The engine
is designed to free the core during a copy: firmware starts an asynchronous
transfer, runs independent compute on other DTCM buffers, and waits for
completion before reading the destination. However, the default simulator
(MPACT) copies all data immediately when a transfer starts (§6), so code that
reads the destination without waiting would pass by accident. To catch missing
waits, the driver defaults to `DEFERRED` issue mode (§4), which holds off the
actual copy until the wait call (`coralnpu_dma_wait_done`).

DMA performance has not been measured for now: MPACT cannot measure overlap (§6),
and no test runs on RTL (§8).

`DmaEngine` is confirmed only in the reference SoC (`CoralNPUChiselSubsystem`).
Core-only integrations (`CoreMiniAxi`) have no engine at `0x40050000`, and
whether a production target includes one is not established yet.

At the C runtime (`crt/coralnpu_dma.h`) level, a single asynchronous tile
transfer looks like this:

```c
#include "crt/coralnpu_dma.h"

// Static: the type's aligned(32) attribute provides the required 32B alignment.
static coralnpu_dma_descriptor_t desc;

int load_tile(uint32_t dtcm_dst, uint32_t ddr_src, uint32_t bytes) {
  int rc = coralnpu_dma_copy_async(dtcm_dst, ddr_src, bytes, &desc);
  if (rc != CORALNPU_DMA_OK) return rc;  // EINVAL or EBUSY: nothing issued
  // Independent compute on other DTCM buffers goes here.
  // Do not read/write dtcm_dst, ddr_src, or desc while in flight.
  rc = coralnpu_dma_wait_done();
  if (rc == CORALNPU_DMA_EHARDWARE) {
    uint32_t code = coralnpu_dma_error_code();  // CORALNPU_DMA_ERR_*
    (void)code;
  }
  // On ETIMEOUT, the caller must call coralnpu_dma_abort() (§5).
  return rc;
}
```

End-to-end, DDR↔DTCM data movement spans two layers in this repository:

1. **Compiler Tile Promotion & Copy Lowering**
   (see [dtcm_tiling.md](dtcm_tiling.md) and §7.2 below): sizes tiles to fit the
   DTCM budget, promotes them to DTCM buffers, and lowers most copies to RVV
   loops.
2. **CRT Driver (`crt/coralnpu_dma.{h,c}`)**: validates `coralnpu_dma_copy_async`
   transfer size/beat alignment and chain length, manages the hardware MMIO
   registers, and implements `DEFERRED` and `IMMEDIATE` issue modes (§4).

**Terms.**

- **M, N, K:** A contraction's output row, output column, and reduction
  dimensions.
- **Split-K:** Tiling the reduction dimension `K` into smaller chunks when full
  `M × K` and `K × N` input tiles do not fit in DTCM at once. The workgroup
  loops over the `K` chunks, loading each input chunk into DTCM and accumulating
  into a single `M × N` buffer in DTCM before writing the final result to DDR.
- **Dispatch ELF:** The bare-metal RISC-V executable linked by
  `CoralNPULinkerTool::linkDynamicLibrary` for one IREE executable. It contains
  the generated kernels, the dispatch ABI entry point (`crt/coralnpu_main.c`),
  and the CRT.
- **Firmware heap:** The region of DTCM between static data and the stack,
  bounded by linker symbols `__heap_start` and `__heap_end`.
- **Zvt:** CoralNPU's matrix-multiply extension (see
  [matrix_codegen.md](matrix_codegen.md)).
- **MPACT:** The functional simulator, and the default execution backend (§6).
- **HAL:** IREE's Hardware Abstraction Layer, the runtime's device driver layer.

---

## 1. Scope: Two Tiers of Data Movement

Data moves at two levels in the system; this document covers only the second:

1. **Host ↔ device and across dispatches.** Managed by IREE's `Flow` and
   `Stream` dialects and the HAL driver. Before each dispatch, the runtime
   stages HAL tensor bindings into DDR (`iree_hal_simulator_issue_dispatch_inline`
   in `runtime/sim/simulator_inline.c`). Out of scope here.
2. **Inside one dispatch, DDR ↔ DTCM.** Managed by **codegen** (Linalg, SCF,
   MemRef). The closest upstream IREE path is on AMD GPUs (ROCDL), which load
   from global memory into on-chip local memory (LDS) with `gather_to_lds` and
   use `ROCDLPrefetchSharedMemoryPass` for pipelining. That path is load-only and
   uses no descriptor engine. `CoralNPULowerDTCMCopies` lowers the copies at this tier.

---

## 2. Hardware & System Architecture

```text
                  Host: ELF load, DTCM mailbox, Core CSR kick
                |
                v
   +-------------------------+
   | AXI4 -> TL-UL Bridge    |
   | (or spi2tlul on FPGA)   |
   +------------+------------+
                |
+---------------|---- CoralNPU Subsystem (~800 MHz) -------------------------+
|               |                                   +-- 256b AXI4 (CDC) --+  |
|               v                                   | ddr_mem             |  |
|  +===================================================================+  |  |
|  |          128-bit TileLink-UL Crossbar (CoralNPUXbar)              |  |  |
|  +=====+===============+==================+==============+======+====+  |  |
|        ^               |                  |              ^      |       |  |
|        | coralnpu_core | coralnpu_device  | sram         | dma  | dma   |  |
|        | (128b master, | (128b slave to   | (128b slave) | host | dev   |  |
|        |  1-outstand.  |  TCMs & CSRs)    v              | 128b | 32b   |  |
|        v  DBus2Axi)    v           +--------------+      v      v       |  |
|  +----------------------------+    | EXTMEM       |   +-------------+   |  |
|  |       CoralNPU Core        |    | (TlulSram)   |   | DMA Engine  |   |  |
|  |     | ibus      ^ dbus     |    | 0x2000_0000  |   | (DmaEngine) |   |  |
|  |     | 128b      | 128b     |    | 4 MB SRAM    |   | 0x4005_0000 |   |  |
|  |     v (1-cyc)   v (1-cyc)  |    +--------------+   | Linked-list |   |  |
|  |  +----------+ +----------+ |    (~10-18 ns CPU;    | 1-outstand. |   |  |
|  |  | ITCM     | | DTCM     | |     ~2.5-5 ns DMA)    +-------------+   |  |
|  |  | 0x0      | | 0x10000  | |  Core CSRs: 0x0003_0000 / 0x0020_0000   |  |
|  |  | 8 KB     | | 32 KB    | |  Core <-> DTCM: 1 cycle,                |  |
|  |  +----------+ +----------+ |                 16 B/cyc (12.8 GB/s)    |  |
|  +----------------------------+                                         |  |
+-------------------------------------------------------------------------|--+
                                                                          |
                +---------------------------------------------------------+
                | Off-chip DRAM bus
                v
   +----------------------------------------------------------------------+
   |            Off-Chip DRAM (DDR)                                       |
   |            0x8000_0000 (up to 2 GB; HAL tensor bindings)             |
   |            NPU CPU / DMA beat: ~75-160 ns                            |
   +----------------------------------------------------------------------+
```

The clock, 12.8 GB/s, DDR, and EXTMEM figures are planning assumptions, not
measurements. FPGA builds in `third_party/coralnpu/fpga/BUILD` run at 30–50 MHz.

### 2.1 DMA Hardware Interface

The hardware specification lives in `third_party/coralnpu/doc/peripherals/dma.md`
(RTL in `DmaEngine` under `third_party/coralnpu/hdl/chisel/src/bus/DmaEngine.scala`,
instantiated in `CoralNPUChiselSubsystem`). Software depends on the following
hardware interface contract:

- **Topology & Reach:** Single-channel linked-list DMA engine with a 128-bit
  TileLink-UL host port (up to 16 B per beat, one outstanding transaction) and
  a 32-bit device CSR port at `0x40050000`. The host port can read and write
  `ITCM`/`DTCM` (`coralnpu_device`), `EXTMEM` (`sram`), `DDR` (`ddr_mem`),
  `rom`, and peripheral FIFOs.
- **Arbitration:** The DTCM has one port with fixed priority. The core always
  wins and never stalls on DMA, so DMA beats and descriptor fetches wait for
  cycles the core leaves free. The core also has priority over DMA at the
  crossbar, including at `ddr_mem` and `sram`.
- **Descriptors (`32 bytes`, 32-byte aligned):** Each descriptor specifies a
  1-D transfer: `src_addr` (`32b`), `dst_addr` (`32b`), `xfer_len` (`24b` byte
  count), `xfer_width` (`3b`, $\log_2(\text{beat bytes}) \in [0, 4]$), flags
  (`src_fixed`, `dst_fixed`, `poll_en`), `next_desc` (`32b` pointer, `0` ends
  the chain), and peripheral flow-control fields (`poll_addr`, `poll_mask`,
  `poll_value`). There is no hardware 2-D stride mode; 2-D strided tiles chain
  one descriptor per row.
- **Completion Notification:** Polling `STATUS.done` only (no interrupts).

**MMIO Registers (`0x40050000`):**

| Offset  | Register      | Access | Interface Contract                                                                                     |
| :------ | :------------ | :----- | :----------------------------------------------------------------------------------------------------- |
| `+0x00` | `CTRL`        | RW     | Bit 0: `enable` (readable); Bit 1: `start` (write-only; self-clearing from idle when `enable=1`, ignored if `enable=0`, latched if written while busy); Bit 2: `abort` (write-only) |
| `+0x04` | `STATUS`      | RO     | Bit 0: `busy`; Bit 1: `done` (level-triggered; cleared by next start from idle or abort); Bit 2: `error`; Bits `[7:4]`: `error_code` |
| `+0x08` | `DESC_ADDR`   | RW     | 32-byte-aligned address of the first descriptor in the chain                                           |
| `+0x0C` | `CUR_DESC`    | RO     | Address of the currently executing descriptor                                                          |
| `+0x10` | `XFER_REMAIN` | RO     | Remaining byte count (`[23:0]`) for the active descriptor                                              |

### 2.2 Memory Map & DTCM Layout

For core integration details, see the
[CoralNPU Integration Guide](https://github.com/google-coral/coralnpu/blob/main/doc/integration_guide.md).
Memory regions are defined in the linker script template
(`MEMORY` in `third_party/coralnpu/toolchain/coralnpu_tcm.ld.tpl`) and
instantiated in `crt/BUILD.bazel` (`coralnpu_tcm_ld` and, for
simulation/verification-only highmem builds, `coralnpu_tcm_highmem_ld`):

| Memory | Default Build            | Highmem Build            | Contents                                                                                      |
| :----- | :----------------------- | :----------------------- | :-------------------------------------------------------------------------------------------- |
| ITCM   | `0x00000000`, 8 KB       | `0x00000000`, 1 MB       | Code, `.rodata`, and init arrays                                                              |
| DTCM   | `0x00010000`, 32 KB      | `0x00100000`, 1 MB       | Static data, firmware `.heap` (dispatch metadata + local memory), and `.stack` at the top     |
| EXTMEM | `0x20000000`, 4 MB       | `0x20000000`, 4 MB       | `.extdata` / `.extbss` (`sram` on the crossbar, reachable by DMA)                             |
| DDR    | `0x80000000`, up to 2 GB | `0x80000000`, up to 2 GB | DMA trace, then HAL tensor bindings (1 GiB window `IREE_HAL_CORALNPU_DDR_BASE`..`IREE_HAL_CORALNPU_DDR_END` in `runtime/sim/simulator_inline.c`) |

> [!NOTE]
> Compiler passes and linker defaults target the 32 KB DTCM layout. Highmem
> (1 MB) builds are used only for simulation and verification and require both
> `--coralnpu-linker-script-path=<...>/coralnpu_tcm_highmem.ld` and
> `--coralnpu-dtcm-size-kb=1024` (as configured in `coralnpu_example_test` in
> `tests/examples/defs.bzl`).

**DTCM Layout & Budget Summary** (`SECTIONS` in `coralnpu_tcm.ld.tpl`,
`iree_hal_simulator_issue_dispatch_inline` in `runtime/sim/simulator_inline.c`,
and `getDTCMTileBudgetBytes` in `CoralNPUTileSizeSelectionUtils.h`):

| DTCM Region (Low → High)          | Default (32 KB)       | Highmem (1 MB)         | Notes & Compiler Flags                                                                                      |
| :-------------------------------- | :-------------------- | :--------------------- | :---------------------------------------------------------------------------------------------------------- |
| 1. Static data (`.data`, `.bss`, `.noinit`) | ~260–270 B (4 KB reserve) | ~260–270 B (4 KB reserve) | Holds dispatch request (`coralnpu_dispatch_request_t`) and `environment`; reserved via `--coralnpu-dtcm-reserve-kb=4` |
| 2. Firmware `.heap`: Push constants & binding tables | `4B × consts + 8B × bindings` (typ. 16–32 B) | `4B × consts + 8B × bindings` (typ. 16–32 B) | Allocated first from `__heap_start` before each dispatch (`binding_ptrs` + `binding_lengths`, up to 64 bindings) |
| 3. Firmware `.heap`: Workgroup local memory (`local_memory`) | **24 KB** (6 pages)   | **1016 KB** (254 pages)| 64-byte aligned, sized in 4 KB pages (`dtcmSizeKb - stackSizeKb - dtcmReserveKb`, rounded down to 4 KB)     |
| 4. `.stack` (`__stack_size__`)    | 4 KB (3 KB alloc max) | 4 KB (3 KB alloc max)  | Top of DTCM (`--coralnpu-stack-size-kb=4`); grows downward with no hardware guard against heap collision    |

---

## 3. The Software Contract (`crt/coralnpu_dma.h`)

`crt/coralnpu_dma.h` locks the register offsets, `coralnpu_dma_descriptor_t`
(verified as 32 bytes via `static_assert`), and the software invariants that
both handwritten callers and compiler codegen must obey:

| #  | Rule                                                                         | Rationale                                                                                                                                                              |
| :- | :--------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1  | Descriptors are 32 bytes and 32-byte aligned                                 | Hardware fetches each descriptor as two 128-bit TileLink beats (`sFetchDesc0`, `sFetchDesc1`).                                                                         |
| 2a | `xfer_len` must be a non-zero multiple of the beat size                      | The 24-bit `remaining` counter (`new_remaining` in `DmaEngine.scala`) stops only at 0; zero length underflows to 16 MB and a non-multiple never reaches 0.             |
| 2b | `src_addr` and `dst_addr` must be beat-aligned                               | Required by TileLink-UL natural alignment and by `DmaEngine.scala`, which shifts the 16-byte lane mask (`sizeMask << addr[3:0]`) and data within a single 128-bit beat without splitting across 16-byte boundaries. |
| 3  | Never modify an in-flight descriptor chain                                   | Hardware fetches `next_desc` entries directly from memory as the chain executes.                                                                                       |
| 4  | Single channel: issue concurrent transfers as one linked chain or serialize  | Hardware has one channel and one outstanding transaction.                                                                                                              |
| 5  | Pair every successful start with **exactly one** wait                        | `STATUS.done` is level-triggered and stays asserted until the next start from idle.                                                                                    |
| 6  | `poll_en` requires `poll_addr != 0`                                          | Required for peripheral FIFO flow control.                                                                                                                             |

- **Rule 2a Overrun Hazard:** A zero or non-beat-multiple `xfer_len` overruns
  the destination buffer until an unmapped address faults (`EHARDWARE`),
  overwriting the top-of-DTCM stack first when targeting DTCM, or timing out
  (`ETIMEOUT`) on long mapped ranges or fixed-address transfers.

**Beat Selection (`coralnpu_dma_choose_width`).** Returns the widest $\log_2$
beat width from `4` (16 B) down to `CORALNPU_DMA_MIN_WIDTH_LOG2` (`2`, i.e.,
4 B) that divides `len`, `src`, and `dst`. Rather than silently falling back to
2 B or 1 B beats (running up to 16× slower), it returns
`CORALNPU_DMA_WIDTH_INVALID`, so a misaligned transfer fails loudly.
`coralnpu_dma_make_len_flags(len, width_log2, flags)` packs the descriptor's
`len_flags` word.

`coralnpu_dma_choose_width` does not check the length range. With 16 B-aligned
addresses, it returns `4` for `len == 0` and `len == 0x01000000`, and
`coralnpu_dma_make_len_flags` then truncates the latter to 0. Only
`coralnpu_dma_copy_async` rejects these lengths, so direct `coralnpu_dma_start`
callers and codegen must ensure `0 < len <= CORALNPU_DMA_LEN_MASK`.

**Return Codes & Fences:**

- **Driver return codes:** `CORALNPU_DMA_OK` (`0`), `CORALNPU_DMA_EBUSY` (`-1`,
  previous chain still in flight; see §4), `CORALNPU_DMA_EINVAL` (`-2`, invalid
  size, alignment, or chain length), `CORALNPU_DMA_ETIMEOUT` (`-3`, poll limit
  exceeded), and `CORALNPU_DMA_EHARDWARE` (`-4`, `STATUS.error` set; inspect
  hardware code `1..4` via `coralnpu_dma_error_code()`). Abort sets code 5
  without `done`, so `coralnpu_dma_wait_done` cannot report it. The RTL is
  authoritative here: `third_party/coralnpu/doc/sw/dma.md` says `done=1`.
- **Memory ordering (`coralnpu_dma_fence`):** Executes `fence iorw, iorw` before
  issuing `CTRL.start` (so descriptor and source stores are visible to the DMA
  engine) and after observing `STATUS.done` (so DMA writes are visible to
  subsequent CPU/RVV/Zvt loads). The fence after `CTRL.abort` orders only the
  core's accesses: a write the engine has already issued can land after it.

---

## 4. The CRT Driver (`crt/coralnpu_dma.c`)

| Function                                        | Core Behavior                                                                                                                          |
| :---------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------- |
| `coralnpu_dma_copy_async(dst, src, size, desc)` | Validates `size` and `busy()`, selects beat width via `coralnpu_dma_choose_width`, populates `*desc`, and calls `coralnpu_dma_start`. |
| `coralnpu_dma_start(desc)`                      | Returns `EBUSY` if busy or `EINVAL` if `coralnpu_dma_walk_chain` fails; fences, records a `START` trace event, and issues or defers.   |
| `coralnpu_dma_wait_done()`                      | Issues any deferred chain after spinning out modelled latency, polls `STATUS.done` up to `CORALNPU_DMA_POLL_LIMIT`, fences, and traces `DONE`. |
| `coralnpu_dma_busy()`                           | Checks software pending state in `DEFERRED` mode or `STATUS.busy` in `IMMEDIATE` mode.                                                 |
| `coralnpu_dma_abort()`                          | Clears software pending state, writes `CTRL.abort`, and fences.                                                                        |
| `coralnpu_dma_error_code()`                     | Extracts hardware error code `STATUS[7:4]`.                                                                                            |

### Core Behaviors & Invariants

- **Busy Guard & Chain Validation (`coralnpu_dma_walk_chain`):**
  - Because hardware latches `CTRL.start` when written while busy (§2.1), both
    `coralnpu_dma_start` and `coralnpu_dma_copy_async` reject concurrent starts
    with `EBUSY` (`copy_async` checks before touching `*desc`, which may still
    be in flight).
  - `EBUSY` means a software-pending chain under `DEFERRED` and `STATUS.busy`
    under `IMMEDIATE`. After `ETIMEOUT`, the `DEFERRED` guard no longer sees the
    engine, so only the caller's `coralnpu_dma_abort` protects it.
  - `coralnpu_dma_walk_chain` walks `next_desc` up to
    `CORALNPU_DMA_MAX_CHAIN` (4096) to accumulate payload bytes and modelled
    cycles. It rejects a NULL head and any chain over the limit with `EINVAL`,
    which catches cycles present at start time, but does not check 32-byte
    descriptor alignment (§3 rule 1) or descriptor fields (§3 rule 2a).
  - `coralnpu_dma_copy_async` requires a non-null, 32-byte-aligned `desc` (it
    writes `*desc` before `coralnpu_dma_start` checks for `NULL`) and is the
    only helper that validates length and beat width; direct
    `coralnpu_dma_start` callers must validate each descriptor (§3, §5).
- **Issue Modes (`CORALNPU_DMA_ISSUE_DEFERRED` vs. `CORALNPU_DMA_ISSUE_IMMEDIATE`):**
  - Because MPACT copies the entire chain synchronously inside the `CTRL.start`
    MMIO write (§6), immediate issue would let a missing `coralnpu_dma_wait_done()`
    pass silently in simulation.
  - Under **`DEFERRED` mode (the default)**, `coralnpu_dma_start` only records
    the chain pointer; `coralnpu_dma_wait_done` spins for a modelled delay
    (`coralnpu_dma_spin`, using `CORALNPU_DMA_FETCH_CYCLES` and
    `CORALNPU_DMA_BEAT_CYCLES`) and then writes `DESC_ADDR` and `CTRL.start`.
    Destination memory stays completely stale until `wait_done()`, catching
    premature reads of the destination buffer (`dma_runtime_nowait_test`).
    Nothing reads the source until then, so overwriting the source or a
    descriptor between start and wait goes **uncaught**.
  - Under **`IMMEDIATE` mode (intended for hardware)**, `coralnpu_dma_start`
    writes `DESC_ADDR` and `CTRL.start` immediately. `TODO(DMA)`: no build
    selects it, so FPGA builds also use `DEFERRED` and have no overlap.
  - **Mode Differences:** The two modes are meant to give the same result when
    callers pair every successful `start()` / `copy_async()` with **exactly one**
    `wait_done()`, and touch no source, destination, descriptor, or driver call
    between them. `busy()`, `error_code()`, and a second start in that window
    depend on the mode. `abort()` ends the pairing, so do not wait after it. An
    extra `wait_done()` returns `OK` under `DEFERRED`, even after `ETIMEOUT`,
    while under `IMMEDIATE` it reports the current `STATUS`. Only `DEFERRED` is
    built and tested, so this is intent, not a verified property.
- **Guest Event Trace (`coralnpu_dma_event`):**
  - Appends 16-byte `coralnpu_dma_trace_t` records (`seq`, `phase`, `bytes`,
    `stamp`) to `CORALNPU_DMA_TRACE_ADDR` (`0x80003000` in DDR, up to
    `CORALNPU_DMA_TRACE_CAPACITY = 256` entries) for post-run inspection on the
    host. `stamp` is 0 because no build defines `CORALNPU_DMA_HAVE_RDCYCLE`.
  - HAL bindings start above this region (`IREE_HAL_CORALNPU_DDR_START`), and
    `iree_hal_simulator_issue_dispatch_inline` zeroes the trace count per
    dispatch (`dma_dispatch_test`).

---

## 5. Using the Driver

Single-buffer transfers use `coralnpu_dma_copy_async` followed by
`coralnpu_dma_wait_done` as shown in the [Overview](#overview).

**Building Multi-Descriptor Chains** (e.g., gathering one descriptor per row of
a 2-D strided tile; `tests/dma/dma_chain_test.c` shows the layout, with
alignment proved by static asserts instead of step 2's check):

1. Place `coralnpu_dma_descriptor_t` descriptors in any DMA-reachable
   memory (DTCM, EXTMEM, or DDR) with **32-byte alignment** (automatic for
   static variables via `aligned(32)`; requires explicit 32-byte alignment in
   workgroup local memory).
2. For each descriptor in the chain:
  - Set `src_addr` and `dst_addr`.
  - Ensure `0 < len <= CORALNPU_DMA_LEN_MASK` (§3) and compute
    `w = coralnpu_dma_choose_width(len, src, dst)`, rejecting the transfer if
    out of range or `w == CORALNPU_DMA_WIDTH_INVALID`.
  - Set `len_flags = coralnpu_dma_make_len_flags(len, w, 0)`, zero the poll
    and `reserved` fields, and set `next_desc` to the next descriptor's address
    (`0` on the final descriptor).
3. Call `coralnpu_dma_start(&chain[0])`, perform independent compute, and call
   `coralnpu_dma_wait_done()` once.

**Error Recovery:**

- **`EBUSY` / `EINVAL`:** Nothing was issued; hardware state is unchanged.
- **`EHARDWARE`:** Chain terminated on a bus error, and the destination may be
  partly written. The engine does not validate descriptors, so an illegal one
  reports only through the overrun in §3 rule 2a. `STATUS` clears when the engine
  accepts the next start, which under `DEFERRED` happens inside the next
  `wait_done()`.
- **`ETIMEOUT`:** The engine may still be active; call `coralnpu_dma_abort()`
  before issuing any new transfer, and do not call `coralnpu_dma_wait_done()`
  again for the aborted chain. Treat the destination as indeterminate.
  `TODO(DMA)`: abort does not drain an outstanding bus transaction, so a late
  response can be read as the next chain's descriptor. Restarting after an
  in-flight abort is unsafe without an engine reset. This comes from reading
  `DmaEngine.scala`; no test covers it.

---

## 6. Simulation Backends

| Execution Backend | DMA Support                                                                                                                                         |
| :---------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------- |
| MPACT             | Modelled synchronously in `DmaMemoryWrapper` (`coralnpu_mpact` in `MODULE.bazel`): copies the full chain on `CTRL.start` write and never sets `STATUS.error` |
| Spike             | **Not mapped.** `CoralNPUSpikeSimulator` (`runtime/sim/spike/simulator.cpp`) maps only TCM, EXTMEM, and DDR; accessing `0x40050000` faults |
| Verilator         | **Not mapped in `CoreMiniAxi`.** `CoreMiniAxiSimulator` (`third_party/coralnpu/hw_sim/core_mini_axi_simulator.cc`) wraps only the core + AXI bridge without `CoralNPUXbar` or `DmaEngine` |
| FPGA              | Real `DmaEngine` RTL when the bitstream is built from `CoralNPUChiselSubsystem`. This repository has no FPGA DMA test                               |

- **Simulator Selection:**
  - The execution backend (`mpact`, `spike`, `verilator`, `fpga`) is selected at
    **model execution time** by `--simulator` or `CORALNPU_SIMULATOR`
    (`runtime/driver/registration/driver_module.c`), not per-dispatch and not by
    the compiler.
  - The compiler does not know which simulator will run a compiled module.
    Generated code does not touch the DMA engine, so compiled modules run on
    every backend. Firmware that calls the driver needs MPACT or FPGA:
    `TODO(DMA)`: Spike needs an MMIO device at `0x40050000`, and Verilator needs
    the full SoC subsystem.
- **Simulator Notes:**
  - MPACT's `DmaMemoryWrapper` ignores `xfer_width`, flags, polling, and
    `CTRL.abort`, and copies nothing when `xfer_len == 0` (where hardware copies
    16 MB), so a passing MPACT run does not enforce §3 rules 2a or 2b.
  - MPACT's cycle counter is an instruction counter, and `DmaMemoryWrapper` does
    not model bus beats, so MPACT cannot measure DMA transfer overlap.
  - On the HAL path (`iree_hal_coralnpu_simulator_backend_dispatch_reused`),
    MPACT and FPGA reuse one simulator instance across dispatches in a process
    until a dispatch fails, so firmware must not assume the DMA CSRs start from
    power-on reset. Of the trace, only the count is reset per dispatch, so older
    records can remain.

---

## 7. Compiler Integration

### 7.1 Linking (`CoralNPULinkerTool`)

`CoralNPULinkerTool::linkDynamicLibrary`
(`compiler/Target/CoralNPULinkerTool.cpp`) links every dispatch ELF with
`--whole-archive libcoralnpu_iree.a libcoralnpu_crt.a --no-whole-archive` and
`--gc-sections`. Because `libcoralnpu_crt.a` is compiled with
`-ffunction-sections -fdata-sections` (`crt/rv32.bzl`, `crt/CMakeLists.txt`),
`--gc-sections` strips `coralnpu_dma_*` and `g_dma` from any dispatch ELF that
does not call the driver.

### 7.2 DTCM Tile Promotion & Workgroup Local Memory

[dtcm_tiling.md](dtcm_tiling.md) owns tile sizing, promotion, and copy lowering,
including what is and is not promoted. These facts matter for DMA:

- **Which copies exist:** `CoralNPUPromoteOperands` promotes the tiles of RVV
  contractions with reuse (`hasContractionReuse`). With default features, a
  contraction that Zvt accepts is not promoted, so its operands stay in memory
  and it has no tile copies for DMA to replace.
- **Where the buffers live:** Promoted tiles use IREE's **workgroup local
  memory** (`#iree_codegen.workgroup_local`).
  `LLVMCPUAssignWorkgroupLocalMemory` packs the buffers and records their byte
  size, and `LibraryBuilder.cpp` rounds it up to `local_memory_pages`. The
  runtime allocates that from the DTCM firmware heap
  (`iree_hal_coralnpu_command_buffer_dispatch`,
  `iree_hal_simulator_issue_dispatch_inline`) and passes `local_memory` to each
  workgroup (`main` in `crt/coralnpu_main.c`).
- **What a workgroup can reuse:** Each workgroup computes one M/N tile and writes
  its buffers before reading them, so generated code does not rely on local
  memory across workgroups. Nothing clears it between them; the runtime zeroes it
  once per dispatch. Only a split-K loop iterates tiles inside the generated
  function.
- **How copies lower:** `CoralNPULowerDTCMCopies`, in the pre-LLVM hook, lowers
  them to loops of RVV loads and stores (`lowerToVectorLoops`). Copies with
  strided rows, sub-byte elements, or no elements keep IREE's lowering
  (`isLowerableDTCMCopy`).
- **What the budget counts:** The tile budget is 24 KB with the default DTCM
  (`getDTCMTileBudgetBytes`). `--coralnpu-dtcm-buffers-per-operand` (default 1)
  multiplies input slices in that estimate, but no pass allocates the extra
  buffers, so higher values only shrink tiles.

`TODO(DMA)`: A synchronous transfer can replace a vector loop at each copy in
`CoralNPULowerDTCMCopies`. Overlap is not a per-copy rewrite: it needs a second
buffer and a loop that issues the next copy while the current tile computes, and
its placement is undecided. Two gaps apply
either way: `DTCMFootprintModel` does not count descriptors (32 B per row), and
1 B and 2 B elements are promoted although the driver's 4 B minimum beat can
reject their copies.

---

## 8. Tests (`//tests/dma`)

Each test below is a standalone bare-metal guest ELF linked against
`libcoralnpu_crt.a` under `coralnpu_tcm.ld` and executed by
`tests/dma/dma_sim_runner.cc` in a fresh MPACT simulator instance:

| Test                      | What It Verifies                                                                                                         | Expected Result Word         |
| :------------------------ | :----------------------------------------------------------------------------------------------------------------------- | :--------------------------- |
| `dma_runtime_test`        | DDR→DTCM and DTCM→DDR transfers, the `EBUSY` single-flight guard, `EINVAL` beat/length validation, and a no-op wait      | `0xD11A0000`                 |
| `dma_runtime_nowait_test` | Issues the same transfer **without** `coralnpu_dma_wait_done()`: verifies all 64 destination words remain stale          | `0xD11B0040` (64 mismatches) |
| `dma_chain_test`          | Gathers an 8-row strided tile (64 B rows, 256 B pitch) using an 8-descriptor chain and checks a guard beat past the tile | `0xD11C0000`                 |
| `dma_double_buffer_test`  | Streams four tiles through a two-slot DTCM ping-pong buffer (wait, prefetch next, compute current) and writes back; checks the schedule, not overlap | `0xD11E0000`                 |

`dma_dispatch_test` instead runs dispatch firmware twice on one simulator through
`iree_hal_simulator_issue_dispatch_inline`, checking that large HAL bindings do
not overlap the trace and that the trace resets.

Not covered: `IMMEDIATE` mode, `coralnpu_dma_abort`, `ETIMEOUT`, `EHARDWARE`, and
any run on RTL.

Compiler DTCM promotion and copy-lowering tests live in `tests/transforms/`
(`tile_size_selection_dtcm.mlir`, `tile_size_selection_footprint.mlir`,
`promote_operands.mlir`, `promote_operands_pipeline.mlir`, and
`lower_dtcm_copies.mlir`).
