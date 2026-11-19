# Synchronous DMA Lowering for DTCM Tile Copies

- **Author(s)**: Khaled Talukder
- **Component Doc(s)**: [dma.md](../dma.md), [tiling_and_promotion.md](../tiling_and_promotion.md)
- **Bug(s)**: None *(Buganizer not searched: the `issues` CLI was unavailable)*

---

## 1. Summary & Motivation

Generated code never calls the DMA driver: every DDR↔DTCM tile copy is an RVV
loop that stalls the core. Double-buffered overlap will later hide copy latency
behind compute, and it needs a compiler-to-driver path first: a CRT ABI, a
lowering, and tests. This change builds that path behind
`--coralnpu-enable-dma`.

Without async DMA, the core copies tiles synchronously, so promoting read-once
ops (`elementwise`, reductions, `matvec`, `vecmat`, `dot`) only adds work
([tiling_and_promotion.md §1](../tiling_and_promotion.md#1-overview)). Today the
compiler therefore promotes only RVV contractions with 2D reuse. Direct Zvt
roots bypass tile copies, while `i32`, `i8→i8`, quantized/dequant, unaligned
`f32`/`bf16`, and hybrid Zvt contractions promote tiles into DTCM.

Once double-buffered async DMA lands, compute-heavy single-use ops (such as
transcendentals or `softmax`, and broader read-once ops if hardware adds
burst/multi-outstanding DDR transfers) can also benefit from DTCM staging.
Because this change keeps copies synchronous and `DEFERRED` mode adds work per
chain ([dma.md](../dma.md) "Terms", §4), it cannot show a speedup yet. We keep
the reuse gate unchanged here and defer single-use promotion to Section 4.

- **Goals**:
  - With `--coralnpu-enable-dma`, promoted `i32` and `f32` matmul model tests
    pass on MPACT and their dispatch ELFs call the CRT.
  - Without the flag, generated code for promoted tile copies does not change.
- **Non-Goals**:
  - Overlap (needs K-loop pipelining and dual-bank descriptors), promoting
    single-use / zero-reuse ops (only pays off with async overlap), 1 B / 2 B
    elements (no provably legal 4 B beat), Spike and core-only Verilator
    ([dma.md §6](../dma.md#6-simulation-backends)), and third-party patches
    (`iree-0006` already preserves whole-tile copies at the pre-LLVM hook).

---

## 2. Proposed Change & Component Doc Delta

Consider compiling a `300×100` × `100×450` `i32` matmul with
`--coralnpu-enable-dma`. Because Zvt does not support `i32` matmuls, the op
lowers on RVV and promotes its tiles into DTCM, including a strided `100×45`
RHS tile from DDR (100 rows of 180 B, strided at 1,800 B in DDR and packed in
DTCM). The flag sets `coralnpu.dma = true` on the executable target. In the
pre-LLVM hook, `CoralNPUPreserveDTCMCopies` shelters the bufferized copy as
`memref.copy` across `GenericVectorization` and `DropVectorUnitDims`, then
`CoralNPULowerDTCMCopiesToDMA` matches it. It converts both endpoints to `index`
byte addresses and emits
`coralnpu_dma_issue_rows(src, dst, 1800, 180, 180, 100)` followed by
`coralnpu_dma_wait()`. (The `2×100` LHS tile is contiguous on both sides, so
its dimensions merge into a single 800 B row with `num_rows = 1`.) Linking
resolves both `hal.import.static` calls from `libcoralnpu_iree.a` into the
8 KB ITCM. On the NPU core, `issue_rows` chains 4 B-beat descriptors in its
64-entry `.bss` pool, flushing the first 64-row batch early before `wait`
transfers the final 36 rows and polls `STATUS.done`.

The CRT contract lives in
[dma.md §3–§4](../dma.md#3-the-software-contract-crtcoralnpu_dmah) and
`crt/coralnpu_dma_dispatch.h`. The compiler contract lives in
[dma.md §7](../dma.md#7-compiler-integration),
[tiling_and_promotion.md §5.2](../tiling_and_promotion.md#52-key-design-details-for-downstream-passes),
`compiler/Target/Utils.h`, `CoralNPUDTCMCopyUtils.h`, and `Passes.td`.

### Key Mechanisms

- **CRT entry points & pool (`crt/coralnpu_dma_dispatch.h`)**: Two functions in
  `libcoralnpu_iree.a` (`coralnpu_dma_issue_rows`, `coralnpu_dma_wait`) build
  and execute chains from a 64-descriptor (2 KB) `.bss` pool inside the 4 KB
  static reserve. `coralnpu_dma_choose_width` enforces
  `0 < len <= CORALNPU_DMA_LEN_MASK`, and coalescing guards against 24-bit
  overflow. Driver errors fail the dispatch via `coralnpu_dispatch_fail` with
  codes below `-0x100`, printed in decimal and hex by `simulator_inline.c`.
- **Uniform pre-LLVM pass order (`beforeLowerToLLVMHook`)**: Both DMA and
  non-DMA targets run `MemrefCopyToLinalg` → `CoralNPUPreserveDTCMCopies` →
  `GenericVectorization(1024)` → `DropVectorUnitDims` →
  `CoralNPULowerDTCMCopiesToDMA` → `CoralNPULowerDTCMCopies`. Sheltering DTCM
  copies as `memref.copy` stops `GenericVectorization` from unrolling them and
  gives `DropVectorUnitDimsPass` (`vector::transferOpflowOpt`) full read/write
  effects, blocking RAW forwarding and WAW dead-store elimination before DMA
  lowering extracts `Pure` byte addresses. `CoralNPULowerDTCMCopies` then
  cleans up `memref.dim` bounds and vector masks before
  `CoralNPULimitLoopUnrolling`
  ([dma.md §7.2](../dma.md#72-dtcm-tile-promotion--workgroup-local-memory)).
- **Address lowering & unroll guards**: `CoralNPULowerDTCMCopiesToDMA` checks
  `hasCoralNPUDma(funcOp)` per function, reconciles mixed static/dynamic shapes
  (`getUnifiedCopyShape`), merges contiguous inner dims, and passes `index`
  byte addresses (`hal.import.static`) so `IREEExpandStridedMetadataPass`
  (`allowSubviewExpansion = false`, `allowUnresolved = false`) erases all DTCM
  `memref.subview`s before `LLVMCPUAssignWorkgroupLocalMemory`
  ([dma.md §7.2](../dma.md#72-dtcm-tile-promotion--workgroup-local-memory)).
  `CoralNPUOptimizeVectorLoops` and recursive `CoralNPULimitLoopUnrolling`
  reject `CallOpInterface` so DMA calls are not duplicated in 8 KB ITCM.

### Invariants & Fallbacks

| Case / Invariant             | Behavior / Rule                                                                                                       |
| :--------------------------- | :-------------------------------------------------------------------------------------------------------------------- |
| Non-`alloca` DTCM copies     | `isLowerableDTCMCopy` checks non-zero extents on both operands and peels views to exclude stack `memref.alloca`.     |
| DMA eligibility              | `isDmaEligibleCopy` requires external↔DTCM direction, folded-chain resolvable roots, and `>= 4 B` power-of-2 types.  |
| Narrow / unresolvable copies | External↔DTCM `< 4 B` (`i8`/`i16`/`bf16`) or control-flow memrefs emit a missed remark and fall back to RVV loops.   |
| DTCM↔DTCM / strided rows     | DTCM-to-DTCM copies use RVV loops without a remark; strided-row or zero-extent copies keep IREE's lowering silently. |
| Target attribute             | `--coralnpu-enable-dma` sets `kCoralNPUDmaAttrName` (`coralnpu.dma`); when absent, DMA lowering is a no-op.          |

### Alternatives Considered

- **Keep RVV copies only**: Overlap needs the ABI and lowering, and FPGA
  measurements can run in parallel.
- **`iree_codegen.ukernel.generic` with memref operands**: The DTCM base buffer
  survives as a non-alloc `workgroup_local` value, which
  `LLVMCPUAssignWorkgroupLocalMemory` rejects.
- **One sync `copy_rows` call / full overlap ABI now**: A single sync call
  cannot chain multiple 2-D slices into one `wait`, while the full overlap ABI
  depends on its schedule (dual-bank / `.extbss` descriptors and
  `coralnpu_dma_dispatch_start`). `issue_rows` + `wait` with a 64-entry sync
  pool keeps multi-slice chaining now and defers dual-bank storage to §4.
- **Backend option (`--coralnpu-fast-math` style)**: Lit tests could not set it
  in target attributes and passes could not query it per function.
- **LLVM intrinsics with inline MMIO**: Wrong memory effects would let LLVM
  reorder loads across the wait.
- **Lower before `DropVectorUnitDimsPass`, fork it, or shelter inside
  `CoralNPULowerDTCMCopies`**: Lowering before `DropVectorUnitDimsPass` breaks
  RAW forwarding and WAW DSE (`extract_aligned_pointer_as_index` is `Pure`).
  Forking duplicates an upstream pass; sheltering inside
  `CoralNPULowerDTCMCopies` couples RVV lowering to DMA remarks. A small
  `CoralNPUPreserveDTCMCopies` pass keeps one linear pipeline and clean owners.
- **Highmem (1 MB) for flagged tests**: Verification-only; hides 8 KB ITCM
  overflows from statically linking the DMA driver.

---

## 3. Implementation & Verification Plan

- [ ] **CL 2a — CRT (`crt/coralnpu_dma.h`, `crt/coralnpu_dispatch.h`,
  `crt/coralnpu_dma_dispatch.{h,c}` (new `.c`), `crt/coralnpu_main.c`,
  `runtime/sim/simulator_inline.c`, `crt/{BUILD.bazel,CMakeLists.txt}`)**:
  Validate `0 < len <= CORALNPU_DMA_LEN_MASK` in `coralnpu_dma_choose_width`.
  Implement both entry points in `libcoralnpu_iree.a` with overflow-safe
  coalescing and a 64-entry `aligned(32)` `.bss` struct (`_Static_assert`ed a
  multiple of 4 B because `crt_section_clear` in `crt.S` traps with `ebreak`
  on unaligned `__bss_end`). Export `coralnpu_dispatch_fail` in
  `crt/coralnpu_dispatch.h`, use `CORALNPU_DISPATCH_ERR_*` in `main`, and log
  `return_code` in decimal and hex in `simulator_inline.c`.
- [ ] **CL 2a — ITCM gate**: `.text` of `coralnpu_dma_dispatch.o` and
  `coralnpu_dma.o` (`llvm-size`) must fit 8 KB minus max `itcm.used`
  (`metrics/runtime/firmware_memory.c`) of promoted matmul ELFs; otherwise
  "Compile Out the Guest Trace" moves ahead of CL 2b.
- [ ] **CL 2a — Tests (new `tests/dma/dma_issue_rows_test.c`,
  `tests/dma/dma_dispatch_runner.cc`, `tests/dma/BUILD.bazel`)**: Check
  `status`, `return_code`, and guest trace records in `after_run`. *Positive*:
  1-descriptor contiguous tile, strided tile, chained back-to-back
  `issue_rows` calls, tile exceeding the 64-entry pool, and zero-size no-op,
  asserting exact trace counts and bytes. *Negative*: misaligned address and
  length `> CORALNPU_DMA_LEN_MASK` returning `CORALNPU_DISPATCH_STATUS_ERROR`
  and `CORALNPU_DISPATCH_ERR_DMA(CORALNPU_DMA_EINVAL, 0)`.
- [ ] **CL 2b — Target fact (`CoralNPUTargetBackend.cpp`,
  `compiler/Target/Utils.{h,cpp}`, `README.md`)**: Record `kCoralNPUDmaAttrName`
  (`coralnpu.dma`) in `getExecutableTarget`, implement `hasCoralNPUDma` in
  `compiler/Target/Utils.cpp`, and note MPACT/FPGA-only execution in `README.md`.
- [ ] **CL 2b — Lowering (`CoralNPUDTCMCopyUtils.cpp` (new),
  `CoralNPUPreserveDTCMCopies.cpp` (new), `CoralNPULowerDTCMCopiesToDMA.cpp`
  (new), `CoralNPULowerDTCMCopies.cpp`, `CoralNPUOptimizeVectorLoops.cpp`,
  `CoralNPULimitLoopUnrolling.cpp`, `CoralNPUTargetBackend.cpp`,
  `compiler/Transforms/{BUILD.bazel,CMakeLists.txt}`)**: Implement shared copy
  predicates and `getUnifiedCopyShape` in `CoralNPUDTCMCopyUtils.cpp`, plus
  `CoralNPUPreserveDTCMCopies` and `CoralNPULowerDTCMCopiesToDMA` per
  `Passes.td`. Update `CoralNPULowerDTCMCopies` to use `getUnifiedCopyShape`
  and clean up `memref.dim` / vector masks. Reject `CallOpInterface` in
  `CoralNPUOptimizeVectorLoops` (`unrollFilterLoops`,
  `unrollAccumulatingInnerLoops`) and recursively in
  `CoralNPULimitLoopUnrolling`. Wire the uniform pre-LLVM order in
  `CoralNPUTargetBackend.cpp` (opening a fresh `pm.nest<func::FuncOp>()` after
  the `ModuleOp` DMA pass) and refresh comments in `CoralNPUPromoteOperands.cpp`.
- [ ] **CL 2b — Tests**:
  - *Unit & Pipeline* (`tests/transforms/lower_dtcm_copies_to_dma.mlir` (new),
    `tests/transforms/lower_dtcm_copies_to_dma_pipeline.mlir` (new),
    `tests/transforms/promote_operands_pipeline.mlir`,
    `tests/transforms/CMakeLists.txt`): Cover rank 1, rank 3 (collapsible, unit
    dim, non-collapsible), mixed static/dynamic `memref.copy`, and partial
    strided DTCM tiles. Verify RAW and WAW ordering through
    `coralnpu-preserve-dtcm-copies` → `iree-codegen-drop-vector-unit-dims` →
    `coralnpu-lower-dtcm-copies-to-dma`, and zero DTCM `subview`s through
    `iree-codegen-expand-strided-metadata` +
    `iree-llvmcpu-assign-workgroup-local-memory`. Add a dynamic-boundary tile to
    `promote_operands_pipeline.mlir`, and assert in
    `lower_dtcm_copies_to_dma_pipeline.mlir` (default `+zvtbase` features) that
    flagged `i32` and `f32` matmuls emit both DMA calls.
  - *Models* (`tests/models/linalg/{defs.bzl,op_tests_{i16,i32,f32}.bzl,CMakeLists.txt}`):
    Add `--coralnpu-enable-dma` `coralnpu_check_test` targets (`_dma` in CMake)
    over existing `generated_{i32,f32}/*_check.mlir` for `matmul_i32`,
    `batch_matmul_i32`, `matmul_f32` `(4,8)(8,4)` / `(300,100)(100,450)`, and
    `dequant_matmul_f32 (22,256)(256,200)`. Check stack peak and vector spills
    on split-K `matmul_i32 (120,256)(256,300)` via
    `--coralnpu-dump-register-allocation-report-format=pretty`.
  - *Negative / Fallback*: Verify `i8`/`bf16` and unresolvable control-flow
    memrefs fall back to RVV with a missed remark; DTCM-to-DTCM,
    `subview(alloca)`, and strided-row copies emit no remark; flagged
    `matmul_i16` passes on RVV; and unflagged `//tests:ci` is unchanged.
- [ ] **Build, Doc & Archive Sync**: Sync `BUILD.bazel`, `CMakeLists.txt`, and
  `refresh_compile_commands`, and run `./scripts/format-code.sh`. CL 2a carries
  `crt/` headers and `dma.md` §2.2/§3/§4/§8 (`dma_issue_rows_test`); CL 2b
  carries compiler headers, the rest of `dma.md`, `tiling_and_promotion.md`,
  `INDEX.md`, and `git mv` to `docs/design/archive/`.

### Implementation Notes & Adaptations

- *(None yet)*

---

## 4. Follow-Up Items (`<component>.md` TODOs / Buganizer `2161972`)

- [ ] **Measure DMA on FPGA** (`Buganizer 2161972` | `P1` | `tests/dma/`):
  Benchmark RVV copy, sync DMA, and ping-pong schedules under `IMMEDIATE` with
  the 32 KB layout (prereq: disable guest tracing and `walk_chain`).
- [ ] **Double-Buffered K Reduction** (`dma.md` — expands existing overlap TODO
  | `CoralNPULowerDTCMCopiesToDMA.cpp`, `crt/coralnpu_dma_dispatch.h`):
  Pipeline K-loops after `DropVectorUnitDimsPass` in `beforeLowerToLLVMHook`
  (avoiding a new IREE patch and `buildLLVMCPUVectorLoweringPipeline`'s
  `transferOpflowOpt`, or use memref-typed async DMA ops). Use a single
  `memref<2x...>` indexed by `iter % 2` (`LLVMCPUAssignWorkgroupLocalMemory`
  forbids `workgroup_local` `iter_args`), add `coralnpu_dma_dispatch_start`,
  and use dual-bank or `.extbss` descriptors with a per-tile K-row cap.
- [ ] **Eliminate Vector-Register Clobbers on DMA Calls** (`dma.md` |
  `CoralNPUTargetBackend.cpp`, `crt/`): Before overlap, link DMA dispatch
  helpers via `hal.import.bitcode` + `noinline` (or `riscv_vector_cc`) to avoid
  256 B/K-chunk RVV accumulator spills across `ilp32` calls.
- [ ] **DMA Beat-Width & Overhead Tuning** (`dma.md`, `tiling_and_promotion.md`
  | `CoralNPUTileSizeSelectionDTCM.cpp`, `CoralNPULowerDTCMCopiesToDMA.cpp`,
  `crt/coralnpu_dma.c`): Enforce 16 B inner tile alignment under DMA, split
  16 B-aligned coalesced transfers into 16 B + tail descriptors, add a
  `>= 64 B` static-byte threshold and adjacent-copy chaining, use
  `__builtin_ctz` in `coralnpu_dma_choose_width`, and skip `walk_chain` in
  `IMMEDIATE` mode.
- [ ] **Promote Single-Use (Zero-Reuse) Ops with Async DMA**
  (`tiling_and_promotion.md` | `CoralNPUTileSizeSelectionDTCM.cpp`): Once async
  overlap lands, relax `hasContractionReuse` under `--coralnpu-enable-dma` for
  compute-heavy read-once ops (e.g. transcendentals, `softmax`) and—if hardware
  adds burst/multi-outstanding DDR—broader read-once ops.
- [ ] **Compile Out the Guest Trace** (`dma.md` | `crt/coralnpu_dma.h`): Drop it
  from dispatch firmware without breaking `dma_dispatch_test` or
  `dma_issue_rows_test` (saves 128 B ITCM and eliminates per-chain DDR reads in
  `IMMEDIATE` mode).
- [ ] **Enforce DMA Hardware Rules in MPACT** (`dma.md` |
  `coralnpu-mpact-*.patch`): Validate 32 B descriptor alignment,
  `0 < xfer_len <= CORALNPU_DMA_LEN_MASK`, `xfer_width <= 4`, and beat
  alignment in `DmaMemoryWrapper`.
- [ ] **Reject DMA Modules on Backends Without DMA** (`dma.md` | `runtime/`):
  Check a dispatch ELF marker before running on Spike.
- [ ] **Explain ITCM Overflow** (`dma.md` | `CoralNPULinkerTool.cpp`): Report
  `--coralnpu-enable-dma` when the 8 KB ITCM overflows at link time.
- [ ] **Widen DMA Coverage** (`tiling_and_promotion.md` |
  `CoralNPUTileSizeSelectionDTCM.cpp`): Zvt contractions, convolutions, `mmt4d`,
  and statically 4 B-aligned 1 B / 2 B tiles.
