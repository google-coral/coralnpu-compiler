# CoralNPU Metrics Architecture (`metrics/`)

This document describes the architecture of the CoralNPU telemetry, compiler
metrics, and workload reporting subsystem under `metrics/`. For CLI flag and
environment variable usage, see [metrics/README.md](../../metrics/README.md) and
[README.md](../../README.md).

---

## 1. Runtime Dispatch Statistics & Memory Footprint (`metrics/runtime/`)

### 1.1 Overview & Boundary Design

Runtime dispatch telemetry is decoupled from the HAL driver (`runtime/driver/`)
and simulator backends (`runtime/sim/`) across two narrow C boundaries:

1. **Device-Level Dispatch Observer (`iree_hal_coralnpu_dispatch_observer_t` in
   `runtime/driver/coralnpu_dispatch_observer.h`)**:
   - Configured on `iree_hal_coralnpu_device_params_t.dispatch_observer`. Driver
     registration (`runtime/driver/registration/driver_module.c`) installs the
     default flag-driven observer from `metrics/runtime/stats_flags.h`.
   - `iree_hal_coralnpu_device_dispatch` (`runtime/driver/coralnpu_device.c`)
     calls `begin_dispatch` before invoking `exec_backend.dispatch`
     (`runtime/driver/coralnpu_exec_backend.h`). In `stats_writer.c`,
     `begin_dispatch` looks up or parses and caches the static ELF memory layout
     for `info->dispatch_image` under `stats->mutex` (`firmware_memory.h`) and
     heap-allocates a per-dispatch collector (`hooks.self`) so concurrent
     callers are thread-safe before the backend acquires
     `iree_hal_coralnpu_simulator_mutex`.
   - When `begin_dispatch` succeeds and the backend returns, the device calls
     `end_dispatch` with the dispatch status code (including failed runs) and
     frees the collector. If `begin_dispatch` fails, its error is logged as a
     warning and the dispatch runs without hooks and without `end_dispatch`.
   - Device parameter validation rejects a non-null `begin_dispatch` when
     `end_dispatch` is `NULL`. Any error from `begin_dispatch` or `end_dispatch`
     is logged as a warning to `stderr` and ignored, and the device passes
     `NULL` hooks to the backend whenever no observer is active.
2. **Simulator-Level Run Hooks (`iree_hal_coralnpu_run_hooks_t` in
   `runtime/driver/coralnpu_run_hooks.h`)**:
   - `iree_hal_simulator_issue_dispatch_inline`
     (`runtime/sim/simulator_inline.c`) invokes `before_run` and `after_run`
     immediately around `coralnpu_simulator_run`. Hooks are not called if ELF
     loading or `.heap` staging fails before the core runs.
   - Each hook receives `iree_hal_coralnpu_run_t`, which provides core
     callbacks (`cycle_count`, `read_mem`, `write_mem`), the mailbox address
     (`dispatch_request_addr`), the `.heap` bump-allocator end address
     (`heap_cursor`), and whether the core halted at `mpause` (`halted`).
   - **Cycle Counter Ordering Invariant on RTL**: On Verilator, `read_mem` and
     `write_mem` advance the simulator cycle counter. Therefore, `before_run` in
     `metrics/runtime/stats_collector.c` reads `cycle_count` last (after any
     stack painting), and `after_run` reads `cycle_count` first (before reading
     the mailbox or scanning the stack).

```text
             +----------------------------------------------------+
             |   driver_module.c (runtime/driver/registration/)   |
             +------------+--------------------------+------------+
                          |                          |
             runs module  |                          | stats_writer_from_flags()
                          v                          v
+----------------------------------+   :   +----------------------------------+
| HAL DRIVER (runtime/driver/)     |   :   | METRICS (metrics/runtime/)       |
|  [coralnpu_device.c]             |  [B1] |  [stats_writer.c]                |
|   1. begin_dispatch(&info) ------+===:==>|   - caches ELF memory layout     |
|   2. exec_backend.dispatch()     |   :   |   - creates dispatch collector   |
|   7. end_dispatch(status) -------+===:==>|   - writes JSON / pretty stats   |
+-----------------+----------------+   :   |                 |                |
                  | 2. dispatch        :   |                 | (per dispatch) |
                  v                    :   |                 v                |
+----------------------------------+   :   |  [stats_collector.c]             |
| SIMULATOR (runtime/sim/)         |  [B2] |   - heap.used, optional stack    |
|  [simulator_inline.c]            |   :   |     paint, start cycle_count     |
|   3. before_run(&run) -----------+===:==>|   - end cycle_count, mailbox     |
|   4. coralnpu_simulator_run()    |   :   |     read, optional stack scan    |
|   6. after_run(&run) ------------+===:==>|                                  |
+-----------------+----------------+   :   +----------------------------------+
                  | 4. run             :
                  v                    :   [B1] coralnpu_dispatch_observer.h
+----------------------------------+   :   [B2] coralnpu_run_hooks.h
| FIRMWARE (crt/)                  |   :
|  [coralnpu_main.c]               |   :
|   5. mcycle / minstret around    |   :
|      workgroup loop -> mailbox   |   :
+----------------------------------+   :
```

### 1.2 Firmware-Measured Kernel Counters (`crt/`)

- **64-bit RV32 CSR Reads (`crt/coralnpu_main.c`)**:
  `coralnpu_read_mcycle()` and `coralnpu_read_minstret()` use the standard
  RISC-V RV32 `mcycleh` / `mcycle` / `mcycleh` (and `minstreth` / `minstret` /
  `minstreth`) retry loop to guard against 32-bit low-word rollover. Before the
  3D workgroup loop, `main()` samples `mcycle` first and `minstret` second;
  after the loop, it samples `minstret` first and `mcycle` second so the
  `minstret` read sequence does not inflate `instret.kernel`.
- **Mailbox Layout (`crt/coralnpu_dispatch.h`)**:
  `coralnpu_dispatch_request_t` (84 bytes, enforced by `static_assert` and by a
  runtime ELF symbol-size check in `simulator_inline.c`) reserves 20 output
  bytes starting at offset 64 (`valid_mask`, `kernel_cycles_lo`,
  `kernel_cycles_hi`, `kernel_instret_lo`, `kernel_instret_hi`). The firmware
  sets `CORALNPU_DISPATCH_VALID_KERNEL_CYCLES` (`1u << 0`) and
  `CORALNPU_DISPATCH_VALID_KERNEL_INSTRET` (`1u << 1`) only after the 3D
  workgroup loop completes; early validation or workgroup execution failures
  call `coralnpu_dispatch_fail`, leaving `valid_mask = 0`.
- **ITCM Footprint Guardrails in `crt/coralnpu_main.c`**:
  `crt/coralnpu_main.o` is linked into every dispatch ELF against an 8 KiB ITCM
  budget. To stay within that budget, `coralnpu_dispatch_fail` is marked
  `__attribute__((noinline))` and writes directly to
  `coralnpu_dispatch_request`. In addition, `coralnpu_read_{mcycle,minstret}`
  are marked `__attribute__((always_inline))` with a `"memory"` clobber to
  prevent out-of-line emission under `-Os` and memory reordering across CSR
  reads, and the zero-initialized 80-byte `environment` struct is placed in
  `.bss` (`DTCM`) rather than `.rodata` (`ITCM`).

### 1.3 ELF Memory Footprint & Stack High-Water Mark (`metrics/runtime/`)

- **ELF Linker Symbol & `PT_LOAD` Parsing (`firmware_memory.{c,h}`)**:
  `iree_hal_coralnpu_firmware_memory_read` looks up 8 CRT linker symbols via
  `iree_hal_coralnpu_executable_find_symbol`
  (`runtime/driver/coralnpu_executable.h`): `__itcm_start`, `__itcm_size`,
  `__dtcm_start`, `__dtcm_size`, `__heap_start`, `__heap_end`, `__stack_start`,
  and `__stack_size` (generated from `@coralnpu_hw//toolchain:coralnpu_tcm.ld.tpl`
  in `crt/BUILD.bazel`, with `--defsym=__stack_size__` set from
  `--coralnpu-stack-size-kb`). It then sums the `p_memsz` overlap
  of `PT_LOAD` segments with ITCM and DTCM. Because the linker script places
  `.heap` and `.stack` inside a DTCM `PT_LOAD` segment without `(NOLOAD)`, their
  byte overlaps are subtracted so `memory.dtcm.used` measures static DTCM
  sections (`.data`, `.bss`, `.sdata`, `.sbss`, and `.noinit`) plus any unusable
  alignment padding before `__heap_start` (preserving
  `dtcm.used + heap.size + stack.size == dtcm.size`).
  `memory.heap.used` is `max(heap_cursor, __heap_start) - __heap_start`, which
  includes push constants, binding tables, and any 4 KiB `workgroup_local_memory`
  pages allocated in DTCM for promoted RVV contraction tiles or Zvt scratch
  tiles.
- **Opt-In Stack Painting & Scanning (`stack_peak.{c,h}`)**:
  Enabled via `--coralnpu_stats_collect=stack` (`CORALNPU_STATS_COLLECT=stack`,
  parsed by `iree_hal_coralnpu_stats_collect_parse` in `stats_writer.h`). Before
  execution, `iree_hal_coralnpu_stack_paint` writes `0xDEADBEEF` in 256-byte
  chunks across the word-aligned stack. When the heap ends at `__stack_start`
  and `heap_cursor` leaves unallocated space below `__stack_start`, painting
  extends up to 256 bytes (`IREE_HAL_CORALNPU_STACK_GUARD_SIZE`) into that
  unused heap guard region. After execution (even if the core did not halt),
  `iree_hal_coralnpu_stack_scan` scans upward from the bottom of the painted
  region to the first modified word to compute the peak stack usage. It sets
  `overflow` to `true` if any guard word was overwritten, or if no guard
  existed and the bottom word of the stack was overwritten.
- **Output Schemas & Verification (`stats_writer.{c,h}`,
  `metrics/tests/runtime/`)**:
  `stats_writer.c` emits `coralnpu.stats.v1` JSON (`cycles.{total,kernel}`,
  `instret.kernel`, `memory.{itcm,dtcm,heap,stack}`) or `pretty` summary tables.
  In `pretty` tables, `itcm(max)`, `data(max)`, `heap(max)`, and `stack(max)`
  report `max(itcm.used)`, `max(dtcm.used)`, `max(heap.used)`, and
  `max(stack.peak)` (with `"!"` appended on overflow). Behavior is verified by
  `//metrics/tests/runtime:stats_test`, `//metrics/tests/runtime:memory_test`,
  and `//metrics/tests/runtime:stack_peak_test`.

---
