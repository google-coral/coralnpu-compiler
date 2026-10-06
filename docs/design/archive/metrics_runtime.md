# Decouple Runtime Dispatch Metrics via Run Hooks and Add Kernel & Memory Telemetry

- **Author(s)**: Shaked Flur
- **Component Doc(s)**: [metrics.md](../metrics.md) *(updated in chained CL 2)*
- **Bug(s)**: None (searched component 564546727; no open issues found)

---

## 1. Summary & Motivation

Currently, the CoralNPU runtime (`runtime/metrics/`,
`runtime/sim/simulator_stats.{c,h}`) couples per-dispatch statistics to the HAL
driver and simulator runner, passing a concrete statistics struct through the
execution backend and reporting only total host-side simulator cycles. That
single counter bundles CRT startup (`_start`, `.bss` zeroing, and mailbox
validation) into kernel time and gives no visibility into retired kernel
instructions or 8 KiB ITCM, 32 KiB DTCM, and 4 KiB stack usage.

This change moves runtime dispatch telemetry into `metrics/runtime/` behind
narrow device-observer and simulator run-hook interfaces. It also extends the
firmware mailbox with kernel-only cycle and instruction counts and adds
per-dispatch TCM, heap, and opt-in stack peak reporting.

- **Goals**:
  - Move runtime statistics, CLI flags, formatting, and tests into
    `metrics/runtime/` and `metrics/tests/runtime/` so only driver
    registration (`runtime/driver/registration/`) depends on `metrics/`,
    keeping the core HAL driver (`runtime/driver/`) and simulator backends
    (`runtime/sim/`) independent of `metrics/`.
  - Report firmware-measured 64-bit `cycles.kernel` and `instret.kernel` around
    the 3D workgroup loop in `coralnpu.stats.v1` JSON and `pretty` tables within
    the 8 KiB ITCM budget.
  - Report per-dispatch static ITCM/DTCM usage, `.heap` bump allocation, and
    opt-in `.stack` peak and overflow (`--coralnpu_stats_collect=stack` /
    `CORALNPU_STATS_COLLECT=stack`) across all simulator backends.
- **Non-Goals**:
  - Compiler-side static performance or tile-grid metrics (separate change).
  - CMake test registration for `stack_peak_test` (tracked in **Section 4**).

---

## 2. Proposed Change & Component Doc Delta

Section 1 of [`metrics.md`](../metrics.md) documents the post-change runtime
telemetry architecture, boundary diagram, and invariants, with boundary
declarations in `runtime/driver/coralnpu_dispatch_observer.h`,
`runtime/driver/coralnpu_run_hooks.h`, `crt/coralnpu_dispatch.h`, and
`metrics/runtime/*.h`:

- **Two-Level Observation Boundary ([`metrics.md` §1.1](../metrics.md))**:
  Replace `on_dispatch` and `out_stats` with `begin_dispatch` / `end_dispatch`
  observer hooks (`runtime/driver/coralnpu_dispatch_observer.h`) that pass
  `before_run` and `after_run` hooks (`runtime/driver/coralnpu_run_hooks.h`) to
  the simulator runner. Observer errors log a warning and run without hooks.
- **RTL Cycle Counter Ordering Invariant ([`metrics.md` §1.1](../metrics.md))**:
  Because host memory accesses advance the Verilator cycle counter, `before_run`
  reads `cycle_count` last (after stack painting) and `after_run` reads
  `cycle_count` first (before reading the mailbox or scanning the stack).
- **Firmware Kernel Counters & ITCM Budget
  ([`metrics.md` §1.2](../metrics.md))**: Extend `coralnpu_dispatch_request_t`
  (`crt/coralnpu_dispatch.h`) from 64 to 84 bytes with a validity mask and
  64-bit `mcycle` and `minstret` deltas sampled around the 3D workgroup loop in
  `crt/coralnpu_main.c`. Keep `coralnpu_main.o` within 8 KiB ITCM by marking
  `coralnpu_dispatch_fail` `noinline` and placing the 80-byte `environment`
  struct in `.bss` (`DTCM`). Failed or non-halted dispatches leave kernel
  counters unset.
- **ELF Memory Footprint & Opt-In Stack Peak
  ([`metrics.md` §1.3](../metrics.md))**: Parse the 8 CRT linker symbols and
  `PT_LOAD` segments (`metrics/runtime/firmware_memory.h`), cached per ELF image
  in `stats_writer.c`, subtracting `.heap` and `.stack` from the DTCM segment so
  `dtcm.used` reflects static data plus any unusable alignment padding before
  `__heap_start` (preserving `dtcm.used + heap.size + stack.size == dtcm.size`).
  `heap.used` reflects the host bump
  cursor (binding tables, push constants, and 4 KiB `workgroup_local_memory`
  pages for DTCM-promoted RVV contractions or Zvt scratch tiles). When
  `--coralnpu_stats_collect=stack` is set, paint the stack (4 KiB by default,
  configured via `--coralnpu-stack-size-kb`) and up to 256 bytes of adjacent
  unallocated heap guard with `0xDEADBEEF` before execution. Scan upward
  afterward (`metrics/runtime/stack_peak.h`) to report `stack.peak` and
  `stack.overflow` even on non-halted faults; missing ELF symbols omit memory
  and stack stats cleanly.

### Alternatives Considered

- **Track kernel cycles and peak stack pointer inside each simulator backend**:
  Avoids DTCM stack-painting traffic on Verilator (~200 ms per dispatch), but
  hardcodes metric schemas into `runtime/sim/`, requires custom PC/register
  hooks in MPACT, Spike, and Verilator, and cannot work on FPGA or silicon.
  Generic memory and cycle run hooks work across all backends while bounding RTL
  overhead by making `--coralnpu_stats_collect=stack` opt-in.
- **Sample `mcycle` and `minstret` via host simulator breakpoints**: Keeps
  `coralnpu_dispatch_request_t` at 64 bytes, but requires symbol lookup of
  internal functions and simulator-specific breakpoint APIs. Sampling RISC-V
  CSRs in `crt/coralnpu_main.c` yields identical counts on MPACT, Spike,
  Verilator, and hardware for 20 extra mailbox bytes.
- **Read memory layout from `simulator_elf_loader.c`
  (`iree_hal_coralnpu_simulator_elf_layout_t`)**: Avoids re-scanning the 6 ELF
  symbols that `simulator_elf_loader.c` already validates, but requires
  expanding `iree_hal_coralnpu_simulator_elf_layout_t` (which only retains heap
  bounds) with TCM/stack bounds and `PT_LOAD` totals and couples `metrics/` to
  `runtime/sim/`. Parsing `info->dispatch_image` once per image in
  `stats_writer.c` keeps run hooks backend-neutral.

---

## 3. Implementation & Verification Plan

### CL 2a: Move Runtime Dispatch Statistics to `metrics/runtime/` Behind Run Hooks
- [x] **Step 1 (`runtime/driver/*`, `runtime/sim/*`)**: Replace `out_stats`,
  `runtime/metrics/dispatch_observer.h`, and `runtime/sim/simulator_stats.{c,h}`
  with `coralnpu_run_hooks.h` and `coralnpu_dispatch_observer.h`.
- [x] **Step 2 (`metrics/runtime/*`, `metrics/README.md`,
  `runtime/driver/registration/*`, `runtime/CMakeLists.txt`,
  `build_tools/cmake/iree_runtime_plugin.cmake`, `README.md`)**: Move
  `runtime/metrics/*` to `metrics/runtime/` and wire registration via
  `stats_flags`.
- [x] **Tests (`metrics/tests/runtime/*`, `tests/BUILD.bazel`)**: Move
  `tests/runtime/*` to `metrics/tests/runtime/*` and verify JSON and `pretty`
  total cycles (`bazel test --config=dev //metrics/tests/runtime:ci`).

### CL 2b: Report Firmware-Measured Kernel Cycles and Retired Instructions
- [x] **Step 3 (`crt/coralnpu_{dispatch.h,main.c}`,
  `runtime/driver/coralnpu_run_hooks.h`, `runtime/sim/simulator_inline.c`,
  `metrics/runtime/*`)**: Extend `coralnpu_dispatch_request_t` to 84 bytes,
  sample 64-bit `mcycle` and `minstret` around the 3D workgroup loop, and report
  `cycles.kernel` and `instret.kernel`.
- [x] **Tests (`metrics/tests/runtime/stats_test.py`)**: *Positive*: Verify
  `0 < cycles.kernel < cycles.total` and `instret.kernel > 0` in JSON and
  `pretty` tables (`bazel test --config=dev //metrics/tests/runtime:stats_test`).
  *Negative*: Verify failed dispatches omit kernel counters when `valid_mask` is
  unset, and verify default-disabled and invalid `--coralnpu_stats_format` runs.

### CL 2c: Add Per-Dispatch ELF Memory Footprint and Stack Peak to Stats
- [x] **Step 4 (`runtime/driver/coralnpu_dispatch_observer.h`,
  `runtime/driver/coralnpu_command_buffer.c`,
  `runtime/driver/coralnpu_run_hooks.h`, `runtime/sim/simulator_inline.c`,
  `metrics/runtime/*`)**: Expose `dispatch_image` on
  `iree_hal_coralnpu_dispatch_info_t` and `write_mem` and `heap_cursor` on
  `iree_hal_coralnpu_run_t`, implement `firmware_memory.{c,h}` (cached per image
  in `stats_writer.c`) and `stack_peak.{c,h}`, and wire opt-in
  `--coralnpu_stats_collect=stack`.
- [x] **Tests (`metrics/tests/runtime/*`)**: *Positive*: Verify TCM/heap/stack
  against `.vmfb` ELF32 sections (`memory_test`), 9 paint/scan cases
  (`stack_peak_test`), and `stack` opt-in and overflow (`bazel test --config=dev
  //metrics/tests/runtime:ci`). *Negative*: Verify unknown
  `--coralnpu_stats_collect` values fail fast and guard overwrites report
  `overflow = true`.
- [x] **Build, Doc & Archive Sync (`metrics/runtime/CMakeLists.txt`,
  `metrics/README.md`, `README.md`, `docs/design/{metrics,INDEX}.md`)**: Sync
  CMake and user docs, verify `metrics.md` and `INDEX.md`, `git mv` this doc to
  `docs/design/archive/metrics_runtime.md`, and run `./scripts/format-code.sh`.

### Implementation Notes & Adaptations
- **CL 2a (`runtime/driver/coralnpu_device.c`)**: Validated in
  `iree_hal_coralnpu_device_check_params` that `end_dispatch` is non-null when
  `begin_dispatch` is provided, and passed `observed ? &hooks : NULL` to
  `exec_backend.dispatch`.
- **CL 2b (`crt/coralnpu_main.c` ITCM Footprint & CSR Barrier)**: Marked
  `coralnpu_dispatch_fail` `__attribute__((noinline))` (writing directly to
  `coralnpu_dispatch_request`), marked `coralnpu_read_{mcycle,minstret}`
  `__attribute__((always_inline))` with a `"memory"` clobber (avoiding mixed
  inline/out-of-line emission under `-Os` and instruction reordering across CSR
  reads), placed the zero-initialized 80-byte `environment` struct in `.bss`
  (`DTCM`) instead of `.rodata` (`ITCM`), and cached `workgroup_count_{x,y,z}`
  in locals to fit the 8 KiB ITCM budget.
- **CL 2c (`metrics/runtime/stats_writer.c`)**: Cached parsed
  `iree_hal_coralnpu_memory_stats_t` per `dispatch_image` on
  `iree_hal_coralnpu_stats_t` under `stats->mutex` in `begin_dispatch` and split
  the `pretty` summary table's DTCM column into `data(max)` and `heap(max)`.

---

## 4. Follow-Up Items (For Reviewer Approval -> Buganizer `564546727`)
- [ ] **Register `stack_peak_test` in CMake and `refresh_compile_commands`**
      (`P3` | `metrics/tests/runtime/`, `BUILD.bazel`): Add `stack_peak_test` to
      CMake and `refresh_compile_commands`.
