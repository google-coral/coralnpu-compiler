# CoralNPU DMA: Architecture, Contract, and Driver

## Overview

CoralNPU executes compute kernels out of fast, tightly coupled on-core memory
(**DTCM**, 32 KB, 1-cycle latency at 16 B/cycle), while model inputs and
outputs live in off-chip **DDR** (~75–160 ns per beat). When generated code
stages tiles in DTCM (§7.2), it copies them mostly with synchronous RVV loops
that stall the core on memory latency. Currently, generated code does not call
the DMA driver; only `tests/dma` calls it.

This document covers the core-local **DMA peripheral** (`dma.DmaCore` in
`third_party/coralnpu/hdl/chisel/src/dma/`) and its CRT driver
(`crt/coralnpu_dma.{h,c}`). Firmware programs memory-mapped registers to start
an asynchronous contiguous copy or strided de-interleave transfer, runs
independent compute on other DTCM buffers, and polls for completion before
reading the destination. Because the default functional simulator (MPACT)
completes each transfer synchronously inside the `CTRL.START` write (§6), the
driver defaults to `DEFERRED` issue mode (§4), which defers the CSR write until
`coralnpu_dma_wait_done()` so premature destination reads observe stale data.

`DmaCore` is instantiated inside `CoreAxi`
(`third_party/coralnpu/hdl/chisel/src/coralnpu/CoreAxi.scala`) when
`p.enableDma = true` (`false` by default in `Parameters.scala`; see §6), mapped
at `0x00032000` on default builds and `0x00202000` on highmem builds. (The
separate legacy SoC descriptor engine `bus.DmaEngine` at `0x40050000` in
`CoralNPUChiselSubsystem` is not used by the CRT or compiler.)

At the C runtime (`crt/coralnpu_dma.h`) level, a single asynchronous tile
transfer looks like this:

```c
#include "crt/coralnpu_dma.h"

int load_tile(uint32_t dtcm_dst, uint32_t ddr_src, uint32_t bytes) {
  int rc = coralnpu_dma_copy_async(dtcm_dst, ddr_src, bytes);
  if (rc != CORALNPU_DMA_OK) return rc;  // EINVAL or EBUSY: nothing issued
  // Independent compute on other DTCM buffers goes here.
  // Do not read/write dtcm_dst or overwrite ddr_src while in flight.
  rc = coralnpu_dma_wait_done();
  if (rc == CORALNPU_DMA_EHARDWARE) {
    uint32_t flags = coralnpu_dma_error_flags();  // CORALNPU_DMA_STATUS_*_ERROR
    (void)flags;
    coralnpu_dma_clear_error();
  }
  return rc;
}
```

End-to-end, DDR↔DTCM data movement spans two layers in this repository:

1. **Compiler Tile Promotion & Copy Lowering**
   (see [tiling_and_promotion.md](tiling_and_promotion.md) and §7.2 below):
   sizes tiles to fit the DTCM budget, promotes them to DTCM buffers, and lowers
   most copies to RVV loops.
2. **CRT Driver (`crt/coralnpu_dma.{h,c}`)**: validates alignment and transfer
   geometry, programs `DmaCore` MMIO registers at `__dma_base`, manages sticky
   hardware error flags, and implements `DEFERRED` and `IMMEDIATE` issue modes
   (§4).

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
   stages HAL tensor bindings into DDR
   (`iree_hal_simulator_issue_dispatch_inline` in
   `runtime/sim/simulator_inline.c`). Out of scope here.
2. **Inside one dispatch, DDR ↔ DTCM.** Managed by **codegen** (Linalg, SCF,
   MemRef). The closest upstream IREE path is on AMD GPUs (ROCDL), which load
   from global memory into on-chip local memory (LDS) with `gather_to_lds` and
   use `ROCDLPrefetchSharedMemoryPass` for pipelining.
   `CoralNPULowerDTCMCopies` lowers the copies at this tier.

---

## 2. Hardware & System Architecture

```text
                Host AXI4 Slave (ELF load, DTCM mailbox, Core CSR kick)
                                      |
+-------------------------------------|---- CoreAxi (~800 MHz) ---------------+
|                                     v                                       |
|  +-----------------------+    +-----------+    +-------------------------+  |
|  |   CoralNPU RV32 Core  |    | BankedItcm|    |       BankedDtcm        |  |
|  |  fetch |      | lsu   |--->| 0x0       |    | 0x0001_0000 (32 KB) /   |  |
|  |  128b  |      | 128b  |    | 8KB / 1MB |    | 0x0010_0000 (1 MB)      |  |
|  +--------+------+-------+    +-----------+    | 5 hosts: AXI RD/WR,     |  |
|                  |                             | Core LSU, DM, DmaCore   |  |
|                  v                             +------------+------------+  |
|     +-------------------------+                             ^               |
|     | Core LSU tlulRouter     |-----------------------------|               |
|     +----+---------------+----+                             |               |
|          | external      | tl_device                        | dtcm          |
|          |               v (0x0003_2000 /     +-------------+---+           |
|          |          +---------+ 0x0020_2000)  |  dmaRouter      |           |
|          |          | DmaCore |               | (ITCM/CSR->err) |           |
|          |          +----+----+               +----+-------+----+           |
|          |               | tl_host (128b,          ^       | external       |
|          |               +--- TlulIdRemapper) -----+       v                |
|          |                   +---------------------------------------+      |
|          +------------------>| ExtArbiter -> TLUL2Axi (axi_master)   |      |
|                              +--------------------+------------------+      |
+---------------------------------------------------|-------------------------+
                                                    | External AXI4
                        +---------------------------+---------------+
                        v                                           v
           +-------------------------+                 +----------------------+
           | EXTMEM (SRAM)           |                 | Off-Chip DRAM (DDR)  |
           | 0x2000_0000 (4 MB)      |                 | 0x8000_0000 (up to   |
           +-------------------------+                 | 2 GB; HAL bindings)  |
                                                       +----------------------+
```

The clock and latency figures are planning assumptions, not measurements. FPGA
builds in `third_party/coralnpu/fpga/BUILD` run at 30–50 MHz.

### 2.1 `DmaCore` Hardware Interface

The hardware specification lives in
`third_party/coralnpu/hdl/chisel/src/dma/README.md` (RTL in `DmaCore.scala`,
`DmaCsr.scala`, `TransposeBuffer.scala`, `CoreAxi.scala`, and
`Parameters.scala`). Software depends on the following hardware contract:

- **Topology & Reach (`CoreAxi.scala`):**
  - `tl_device` (CSR slave, 4 KB aperture at `csrRegion.memStart + 0x2000`:
    `0x00032000` default, `0x00202000` highmem) is routed **only** from the
    CoralNPU scalar core's LSU (`tlulRouter`), not from the external host
    `axi_slave` or instruction fetch.
  - `tl_host` (128-bit TileLink-UL master, `DLENB = 16` bytes per beat) connects
    through `dmaRouter` directly to `BankedDtcm` (`tl_h(4)`) and, via
    `TlulIdRemapper` + `ExtArbiter` + `TLUL2Axi`, to external AXI (`EXTMEM` at
    `0x20000000` and `DDR` at `0x80000000`).
  - Accesses by `tl_host` to **ITCM** (including `.rodata` constants!), Core
    **CSR** (`0x00030000` / `0x00200000`), or **DMA CSR** return a TileLink
    error (`inError`) and set `STATUS.xfer_error`.
- **Two Transfer Modes (`XFER_CFG.n_areas`):**
  - **Memcopy (`n_areas == 1`, reset default):** Streams `ceil(len / 16)`
    16-byte beats through the 16-slot `DmaBypassBuffer` (`Get` and `Put`
    requests overlap). Valid `LEN_FLAGS.xfer_width` (`0..4`; `> 4` still raises
    `cfg_error`) is unused by the data path (`gMax = 4` forced in hardware),
    and `lastMask` masks the final beat when `len % 16 != 0` so trailing bytes
    past `dst + len` are untouched.
  - **Strided de-interleave (`n_areas == 2..16`):** Splits interleaved elements
    of `2^xfer_width` bytes (`xfer_width` in `0..4`, i.e., 1..16 bytes) across
    `n_areas` output planes via `TransposeBuffer`. Software writes
    `AREA_SIZE = len / n_areas`; plane `j` lands at `dst + j * row_pitch` with
    `row_pitch = ceil(area_size / 16) * 16`, and bytes
    `area_size .. row_pitch - 1` of every plane are masked off (`maskAll`).
  - **No Descriptor Chaining or 2-D Source Stride:** Hardware holds one active
    transfer with no queue or linked descriptors. Gathering a 2-D sub-matrix
    with arbitrary source row pitch (`src_pitch > row_bytes`) issues one 1-D
    memcopy per row.
- **Completion & Errors:** Polling `STATUS` only (no interrupts; `CTRL[2]`
  abort and `LEN_FLAGS[29:27]` fixed/poll modes are unimplemented). On error,
  `STATUS.done` stays `0`, `STATUS.error` (`bit 2`) rises alongside the specific
  cause bit (`[5:3]`), and error bits remain sticky until cleared by
  `CTRL.CLEAR_ERROR`.

**MMIO Registers (`0x00032000` default / `0x00202000` highmem; 32-bit aligned
accesses only — offsets outside `0x00..0x1C` return a TileLink error and fault
the core, and MPACT also faults on sub-word or misaligned accesses):**

| Offset  | Register    | Access | Interface Contract                                                                                               |
| :------ | :---------- | :----- | :--------------------------------------------------------------------------------------------------------------- |
| `+0x00` | `CTRL`      | RW     | `[0]` `ENABLE` (only bit returned on reads); `[1]` `START` (self-clearing when `ENABLE=1` and idle); `[2]` `ABORT` (reserved); `[3]` `CLEAR_ERROR` (clears sticky errors; combinable with `ENABLE \| START`) |
| `+0x04` | `STATUS`    | RO     | `[0]` `busy`; `[1]` `done` (clean completion only; cleared on next `START` request); `[2]` `error` (`\|` of `[5:3]`); `[3]` `align_error`; `[4]` `cfg_error`; `[5]` `xfer_error` |
| `+0x08` | `SRC_ADDR`  | RW     | 16-byte-aligned source address (frozen while `busy=1`)                                                           |
| `+0x0C` | `DST_ADDR`  | RW     | 16-byte-aligned destination address of plane 0 (frozen while `busy=1`)                                           |
| `+0x10` | `LEN_FLAGS` | RW     | `[23:0]` `xfer_len` (source bytes); `[26:24]` `xfer_width` ($\log_2$ element bytes `0..4`, strided only); `[31:27]` must be `0` (`[29:27] != 0` raises `cfg_error`) |
| `+0x14` | `XFER_CFG`  | RW     | `[7:0]` `n_areas` (`1..16`; reset `1` = memcopy)                                                                 |
| `+0x18` | `AREA_SIZE` | RW     | `[23:0]` bytes per plane (`len / n_areas`), used when `n_areas > 1` (reset `0`)                                  |
| `+0x1C` | `PASS_REM`  | RO     | Remaining transpose passes (`0` in memcopy bypass; debug only)                                                   |

### 2.2 Memory Map & DTCM Layout

For core integration details, see the
[CoralNPU Integration Guide](https://github.com/google-coral/coralnpu/blob/main/doc/integration_guide.md).
Memory regions are defined in `Parameters.scala` and
`third_party/coralnpu/toolchain/coralnpu_tcm.ld.tpl`, and instantiated in
`crt/BUILD.bazel` and `crt/CMakeLists.txt` (`coralnpu_tcm.ld` and
`coralnpu_tcm_highmem.ld`, which derive `__dma_base` as `<csr_origin> + 0x2000`
unless overridden via `DEFINED(__dma_base)`):

| Region    | Default Build            | Highmem Build            | Contents & DMA Reachability                                                                   |
| :-------- | :----------------------- | :----------------------- | :-------------------------------------------------------------------------------------------- |
| ITCM      | `0x00000000`, 8 KB       | `0x00000000`, 1 MB       | Code, `.rodata`, and init arrays (**unreachable** by `DmaCore` `tl_host` → `xfer_error`)      |
| DTCM      | `0x00010000`, 32 KB      | `0x00100000`, 1 MB       | Static data, firmware `.heap`, and `.stack` (**reachable** via `BankedDtcm`)                  |
| Core CSR  | `0x00030000`, 4 KB       | `0x00200000`, 4 KB       | Host/debug control registers (**unreachable** by `DmaCore` `tl_host` → `xfer_error`)          |
| `DmaCore` | `0x00032000`, 4 KB       | `0x00202000`, 4 KB       | `DmaCore` MMIO CSRs (`__dma_base`; core LSU access only)                                      |
| EXTMEM    | `0x20000000`, 4 MB       | `0x20000000`, 4 MB       | `.extdata` / `.extbss` (**reachable** via external AXI)                                       |
| DDR       | `0x80000000`, up to 2 GB | `0x80000000`, up to 2 GB | DMA trace, then HAL tensor bindings (`runtime/sim/simulator_inline.c`; **reachable** via AXI) |

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

`crt/coralnpu_dma.h` defines register offsets relative to `coralnpu_dma_base()`
(`__dma_base`), geometry helpers, and the invariants callers and compiler
codegen must obey (`third_party/coralnpu/hdl/chisel/src/dma/README.md` §2.6):

| # | Rule                                                                                          | Hardware Consequence if Violated                                                                                  |
| :- | :-------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------- |
| 1 | `SRC_ADDR` and `DST_ADDR` must be 16-byte aligned (`(addr & 0xF) == 0`)                       | `STATUS.align_error` (`bit 3`); transfer does not start (`len` itself may be any byte count in memcopy)           |
| 2 | `0 < len <= 0x00FFFFFF`; for `n_areas > 1`, `len % (n_areas << elem_width_log2) == 0`         | `len == 0` or `AREA_SIZE == 0` sets `cfg_error`; `len > 24b`, non-group-multiple `len`, or wrong non-zero `AREA_SIZE` silently corrupts data |
| 3 | `elem_width_log2 <= 4`, `1 <= n_areas <= 16`, and `LEN_FLAGS[31:27] == 0`                     | `STATUS.cfg_error` (`bit 4`, except reserved `LEN_FLAGS[31:30]`, which hardware ignores); transfer does not start |
| 4 | Strided source readable up to `ceil(area_size / 16) * n_areas * 16` bytes (≤240 B over-read)  | `STATUS.xfer_error` (`bit 5`) if final-pass whole-beat over-read crosses into unmapped memory                     |
| 5 | Strided destination sized `n_areas * coralnpu_dma_row_pitch(area_size)`                       | Undetected buffer overrun or misread plane offsets                                                                |
| 6 | Source and destination must reside in DTCM, EXTMEM, or DDR (never ITCM / `.rodata` or CSRs)   | `STATUS.xfer_error` (`bit 5`) on `dmaRouter` TileLink error                                                       |
| 7 | Clear sticky errors (`CTRL.CLEAR_ERROR`) before starting, and pair each start with one wait   | Uncleared error causes `START` to be ignored; concurrent CSR writes while `busy=1` are ignored                    |

**Helpers, Return Codes, & Fences:**

- **Address & Geometry Helpers:**
  - `coralnpu_dma_base()` returns `(uint32_t)(uintptr_t)__dma_base` on RISC-V
    (`0x00032000` or `0x00202000`) and `CORALNPU_DMA_DEFAULT_BASE` on host.
  - `coralnpu_dma_make_len_flags(len, elem_width_log2)` packs `LEN_FLAGS`.
  - `coralnpu_dma_row_pitch(area_size)` computes `ceil(area_size / 16) * 16`.
- **Driver Return Codes:** `CORALNPU_DMA_OK` (`0`), `CORALNPU_DMA_EBUSY` (`-1`,
  previous transfer not yet awaited or `STATUS.busy` still asserted),
  `CORALNPU_DMA_EINVAL` (`-2`, invalid alignment, length, width, or `n_areas`),
  `CORALNPU_DMA_ETIMEOUT` (`-3`, `CORALNPU_DMA_POLL_LIMIT + beats *
  CORALNPU_DMA_POLLS_PER_BEAT` exceeded), and `CORALNPU_DMA_EHARDWARE` (`-4`,
  `STATUS.error` set; inspect `CORALNPU_DMA_STATUS_{ALIGN,CFG,XFER}_ERROR` with
  `coralnpu_dma_error_flags()`).
- **Memory Ordering (`coralnpu_dma_fence`):** Executes `fence iorw, iorw` before
  issuing `CTRL.START` (ordering source stores before the device write) and
  after observing terminal `STATUS` (ordering the device read before destination
  loads).

---

## 4. The CRT Driver (`crt/coralnpu_dma.c`)

| Function                                                                     | Core Behavior                                                                                                                               |
| :--------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------ |
| `coralnpu_dma_copy_async(dst, src, size_bytes)`                              | Calls `coralnpu_dma_deinterleave_async(dst, src, size_bytes, 0, 1)` for a contiguous 1-D copy (with hardware tail-beat masking).            |
| `coralnpu_dma_deinterleave_async(dst, src, size_bytes, width_log2, n_areas)` | Validates alignment/shape and `!busy()`, fences, records a `START` trace event, and issues (`IMMEDIATE`) or records (`DEFERRED`) the job.   |
| `coralnpu_dma_wait_done()`                                                   | No-op (`OK`) when idle; in `DEFERRED` mode spins modelled beat cycles and issues CSRs; polls `STATUS` until `!busy && (done \|\| error)`.   |
| `coralnpu_dma_busy()`                                                        | Returns nonzero while a started job has not completed `wait_done()` or `STATUS.busy` is set.                                                 |
| `coralnpu_dma_error_flags()`                                                 | Returns `STATUS & CORALNPU_DMA_STATUS_ERROR_FLAGS_MASK` (`ALIGN_ERROR \| CFG_ERROR \| XFER_ERROR`).                                         |
| `coralnpu_dma_clear_error()`                                                 | Preserves `CTRL.ENABLE`, writes `CTRL.CLEAR_ERROR` to clear sticky error flags, and fences.                                                 |

### Core Behaviors & Invariants

- **Single-Flight State Machine (`DMA_STATE_{IDLE,DEFERRED,ISSUED}`):**
  - Every successful `copy_async` / `deinterleave_async` transitions `g_dma` out
    of `DMA_STATE_IDLE` until `coralnpu_dma_wait_done()` observes completion, so
    unpaired starts return `CORALNPU_DMA_EBUSY` in both `DEFERRED` and
    `IMMEDIATE` modes, and an idle `wait_done()` returns `CORALNPU_DMA_OK` in
    both modes.
  - On `ETIMEOUT`, `g_dma.state` stays `DMA_STATE_ISSUED`: `busy()` continues to
    block new transfers and a retry `wait_done()` polls `STATUS` without
    re-issuing `CTRL.START`.
- **CSR Programming & Sticky Error Handling (`coralnpu_dma_issue`):**
  - Always writes `XFER_CFG` (`n_areas`) so a contiguous `copy_async` after a
    strided transfer resets `n_areas` back to `1`, and writes `AREA_SIZE` when
    `n_areas > 1`.
  - Writes `CTRL_ENABLE | CTRL_START` without `CTRL_CLEAR_ERROR`, honoring RTL
    Requirement 8: if a sticky error was left uncleared, hardware ignores
    `START`, preserves the original error cause in `STATUS`, and `wait_done()`
    immediately returns `CORALNPU_DMA_EHARDWARE` until software explicitly calls
    `coralnpu_dma_clear_error()`.
  - `coralnpu_dma_wait_done()` scales its poll budget as
    `CORALNPU_DMA_POLL_LIMIT + beats * CORALNPU_DMA_POLLS_PER_BEAT` and polls
    until `STATUS.busy` clears with `STATUS.done` or `STATUS.error` set because
    on a bus `xfer_error`, `STATUS.error` asserts before `DmaFillEngine` and
    `DmaDrainEngine` finish draining outstanding TileLink beats.
- **Issue Modes (`DEFERRED` vs. `IMMEDIATE`):**
  - Under **`DEFERRED` mode (the default)**, start calls store `pending_job` and
    compute `passes * n_areas * CORALNPU_DMA_BEAT_CYCLES`; `wait_done()` spins
    that delay and then writes the CSRs, keeping destination memory stale until
    `wait_done()` (`dma_runtime_nowait_test`). Overwriting the source buffer
    between start and wait is not caught in `DEFERRED` mode.
  - Under **`IMMEDIATE` mode**, start calls invoke `coralnpu_dma_issue`
    immediately for `CoreAxi` hardware builds elaborated with
    `enableDma = true`. `TODO(DMA)`: select `IMMEDIATE` mode for
    `enableDma = true` hardware/Verilator builds once async DMA codegen lands.
- **Guest Event Trace (`coralnpu_dma_event`):**
  - Appends 16-byte `coralnpu_dma_trace_t` records (`seq`, `phase`, `bytes`,
    `stamp`) to `CORALNPU_DMA_TRACE_ADDR` (`0x80003000` in DDR, up to
    `CORALNPU_DMA_TRACE_CAPACITY = 256` entries). HAL bindings start above this
    region (`runtime/sim/simulator_inline.c`).

---

## 5. Using the Driver

- **Contiguous 1-D Copy:** Call `coralnpu_dma_copy_async(dst, src, bytes)` with
  16-byte-aligned `dst` and `src` and `1 <= bytes <= 0x00FFFFFF`, perform
  independent work, and call `coralnpu_dma_wait_done()`.
- **Strided De-Interleave (e.g., RGB / multi-channel unpack):**
  1. Allocate `src` (16-byte aligned) with readable padding up to
     `ceil((bytes / n_areas) / 16) * n_areas * 16` bytes (§3 rule 4).
  2. Allocate `dst` (16-byte aligned) of
     `n_areas * coralnpu_dma_row_pitch(bytes / n_areas)` bytes; plane `j` begins
     at `dst + j * row_pitch` (§3 rule 5).
  3. Call `coralnpu_dma_deinterleave_async` (`dst`, `src`, `bytes`,
     `elem_width_log2`, `n_areas`) followed by `coralnpu_dma_wait_done()`.
- **2-D Strided Row Gather (`src_pitch > row_bytes`):** Because `DmaCore` has no
  arbitrary source-stride register or descriptor chain, gather 2-D tiles with a
  per-row loop of
  `coralnpu_dma_copy_async(dst + r * dst_pitch, src + r * src_pitch, row_bytes)`
  and `coralnpu_dma_wait_done()` (`tests/dma/dma_chain_test.c`), requiring each
  row's `src` and `dst` start address to be 16-byte aligned.

**Error Recovery:**

- **`EBUSY` / `EINVAL`:** Nothing was issued; hardware state is unchanged.
- **`EHARDWARE`:** Inspect `coralnpu_dma_error_flags()`. For `ALIGN_ERROR` or
  `CFG_ERROR`, the destination was not modified; for `XFER_ERROR`, the
  destination may be partially written. Call `coralnpu_dma_clear_error()` before
  issuing the next transfer; otherwise subsequent transfers leave the error
  latched and return `CORALNPU_DMA_EHARDWARE` without modifying destination
  memory.
- **`ETIMEOUT`:** `DmaCore` does not implement `CTRL.abort`, so a timed-out
  transfer on hardware requires a core/subsystem reset before reusing DMA.

---

## 6. Simulation Backends

| Execution Backend | DMA Support                                                                                                                                                                     |
| :---------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| MPACT             | Modelled synchronously in `DmaMemoryWrapper` (`coralnpu-mpact-0003-dma-core.patch`): maps `0x00032000` / `0x00202000`, tail masking, strided de-interleave, sticky `align`/`cfg`/`xfer` errors, invalid-offset (and sub-word/misaligned) access faults, and per-run CSR reset |
| Spike             | **Not mapped.** `CoralNPUSpikeSimulator` (`runtime/sim/spike/simulator.cpp`) maps `[0x00000000, 0x00200000)` as plain RAM (`0x00032000` reads `STATUS = 0` and hits `ETIMEOUT` without copying; `0x00202000` faults) |
| Verilator         | **Not mapped in `hw_sim` at `HEAD`.** `CoreMiniAxiSimulator` (`third_party/coralnpu/hw_sim/BUILD`) currently elaborates `VmeCoreMiniAxi` / `VmeCoreMiniHighmemAxi` with `enableDma = false` (`DmaCore` RTL is exercised in `core_mini_dma_axi_model` / `core_mini_dma_axi_sim_cocotb` in `third_party/coralnpu/tests/cocotb/BUILD`) |
| FPGA              | Real `DmaCore` RTL when `CoreAxi` is elaborated with `enableDma = true` (current Nexus bitstreams in `third_party/coralnpu/fpga/BUILD` build `CoralNPUChiselSubsystem` with `enableDma = false`). No FPGA DMA test |

- **Simulator Notes:**
  - MPACT's cycle counter counts instructions and `DmaMemoryWrapper` completes
    synchronously on `CTRL.START`, so MPACT cannot measure DMA/compute overlap.
    `TODO(DMA)`: wire `--enableDma=True` into `VmeCoreMini*Axi` /
    `CoreMiniAxiSimulator` in `third_party/coralnpu/hw_sim/` for cycle-accurate
    Verilator bus contention and overlap measurements.
  - On the HAL path (`iree_hal_coralnpu_simulator_backend_dispatch_reused`),
    MPACT reuses one simulator instance across dispatches; `MpactSimulator::Run`
    resets `DmaMemoryWrapper` CSR state per run and latches whether the loaded
    TCM image wrote outside default 8 KB ITCM / 32 KB DTCM so `DmaCore` maps at
    `0x00032000` (default) or `0x00202000` (highmem) without intercepting
    highmem ITCM (`0x00032000`).

---

## 7. Compiler Integration

### 7.1 Linking (`CoralNPULinkerTool`)

`CoralNPULinkerTool::linkDynamicLibrary`
(`compiler/Target/CoralNPULinkerTool.cpp`) links every dispatch ELF with
`--whole-archive libcoralnpu_iree.a libcoralnpu_crt.a --no-whole-archive` and
`--gc-sections`. Both `coralnpu_tcm.ld` and `coralnpu_tcm_highmem.ld` define
`__dma_base`, and `--gc-sections` strips `coralnpu_dma_*` and `g_dma` from any
dispatch ELF that does not call the driver.

### 7.2 DTCM Tile Promotion & Workgroup Local Memory

[tiling_and_promotion.md](tiling_and_promotion.md) owns tile sizing, promotion,
and copy lowering. These facts matter for DMA lowering:

- **Which copies exist:** `CoralNPUPromoteOperands` promotes tiles of RVV
  contractions with reuse (`hasContractionReuse`). Contractions lowered to Zvt
  stay in memory by default.
- **Where the buffers live:** Promoted tiles use `#iree_codegen.workgroup_local`
  memory allocated from the 64-byte-aligned DTCM firmware heap
  (`iree_hal_coralnpu_command_buffer_dispatch`,
  `iree_hal_simulator_issue_dispatch_inline`). Note that `.rodata` constants
  live in ITCM (§2.2), so constant buffers in `.rodata` cannot be read by
  `DmaCore` (`xfer_error`); HAL bindings in DDR (`0x80000000`) and `.extdata` in
  EXTMEM (`0x20000000`) are DMA-reachable.
- **How copies lower today:** `CoralNPULowerDTCMCopies` lowers eligible copies
  to RVV load/store loops (`lowerToVectorLoops`).

`TODO(DMA)`: Lowering DTCM copies to `DmaCore` requires no DTCM descriptor
budget (`0` descriptor bytes) and supports arbitrary non-16B-multiple transfer
lengths via hardware tail masking, provided both `src` and `dst` start addresses
(for every row of a 2-D strided gather) are 16-byte aligned and the source does
not reside in ITCM. True overlap still requires double buffering in DTCM and
`CORALNPU_DMA_ISSUE_IMMEDIATE`.

---

## 8. Tests (`//tests/dma`)

Each standalone test below links against `libcoralnpu_crt.a` and executes in
`tests/dma/dma_sim_runner.cc` on MPACT:

| Test                       | What It Verifies                                                                                                                           | Expected Result Word         |
| :------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------- |
| `dma_runtime_test`         | Live `__dma_base` (`0x00032000`), DDR↔DTCM copies, 22 B tail masking, `EBUSY`/`EINVAL`, sticky `XFER`/`ALIGN`/`CFG` blocking + clear, CSR access faults, idle wait | `0xD11A0000`                 |
| `dma_runtime_highmem_test` | Same suite linked with `coralnpu_tcm_highmem.ld`, verifying `__dma_base` at `0x00202000` and non-interception of highmem ITCM `0x00032000` | `0xD11A0000`                 |
| `dma_runtime_nowait_test`  | Issues a copy **without** `coralnpu_dma_wait_done()`: verifies all 64 destination words remain stale under `DEFERRED` mode                 | `0xD11B0040` (64 mismatches) |
| `dma_chain_test`           | 8-row 2-D strided gather, 3-plane hardware de-interleave across `elem_bytes = 1, 2, 4, 8, 16`, memcopy ignoring `xfer_width`, and `XFER_CFG` reset | `0xD11C0000`                 |
| `dma_double_buffer_test`   | Streams four tiles through a two-slot DTCM ping-pong buffer (wait, prefetch next, compute current) and writes back to DDR                  | `0xD11E0000`                 |
| `dma_timeout_test`         | Overrides `__dma_base` to SRAM via `--defsym`: verifies beat-scaled `ETIMEOUT`, `EBUSY` while timed out, and retry `wait_done()` without re-issuing `START` | `0xD11F0000`                 |

`dma_dispatch_test` runs default and highmem dispatch firmware across four
dispatches on one simulator through `iree_hal_simulator_issue_dispatch_inline`,
checking DDR trace non-overlap/reset, per-dispatch `DmaCore` CSR reset after an
uncleared error, multi-segment highmem ITCM preservation, and `highmem →
default` layout switching.

Compiler DTCM promotion and copy-lowering tests live in `tests/transforms/`
(`tile_size_selection_dtcm.mlir`, `tile_size_selection_footprint.mlir`,
`promote_operands.mlir`, `promote_operands_pipeline.mlir`, and
`lower_dtcm_copies.mlir`).
