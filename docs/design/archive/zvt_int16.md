# Zvt INT16 Matrix Codegen Support

- **Author(s)**: Lun Dong
- **Component Doc(s)**: [matrix_codegen.md](../matrix_codegen.md)
- **Bug(s)**: N/A

---

## 1. Summary & Motivation

Multiplying two signed `i16` values produces up to 31-bit magnitudes
(`(-32768)^2 = 2^30`), so widening 16-bit contractions and convolutions
(`i16 x i16 -> i32`, such as `StableHLO`/JAX ops with
`preferred_element_type=jnp.int32` or quantized `i16` and `Linalg` kernels)
accumulate in `i32`, matching Zvt's 32-bit tile accumulators (`TEW = 32`). While
`i8 x i8 -> i32` already lowers to Zvt `vtmms.tvv`, `i16 x i16 -> i32` currently
falls back to CoralNPU's RVV pipeline (`12.8 GOPS` at `0.8 GHz`, 8 MACs/cycle at
`VLEN = 128`). CoralNPU's 64-PE Zvt array retires up to 128 INT16 MACs/cycle
(`204.8 GOPS` PE-array peak at `0.8 GHz`, 4 cycles per 512-MAC `TK = 2`
`vtmms.tvv`). We extend that existing Zvt tiling, roofline placement, and
microkernel pipeline to signed `INT16 -> INT32` operations—combining `BF16 ->
FP32`'s 16-bit tile geometry with `INT8 -> INT32`'s signed `vtmms.tvv`
accumulation—and enable 16-bit `vtmms.tvv` across MPACT, Spike, and Verilator.

- **Goals**:
  - Lower signed `i16 x i16 -> i32` contractions (`matmul`, `batch_matmul`,
    `16x16x1` single-tile `mmt4d`, generic contractions) and spatial
    convolutions ($M \ge 3, N \ge 8$) through CoralNPU's existing Zvt pipeline
    across full and partial tiles.
  - Enable `SEW = 16, TWIDEN = 2` (`TEW / SEW = 32 / 16`) signed `vtmms.tvv`
    execution across MPACT, Spike, and Verilator RTL (`third_party/coralnpu`)
    and verify all three in `//tests:ci`.
- **Non-Goals**:
  - Non-widening `i16 x i16 -> i16` contractions: a single `i16 * i16` product
    overflows `i16` and Zvt tiles are 32-bit (`TEW = 32`), so `i16 -> i16` stays
    on RVV.
  - Unsigned `ui16 -> i32` (`arith.extui` / `vtmmu.tvv`), mixed-width (`i16 x
    i8`), or `i48` TOSA contractions: 16-bit ML quantization specs use signed
    `int16`, and TOSA `int16` emits `i48 -> i64` accumulators (or `i16 x i8`
    convolutions), so these stay on RVV/host.
  - Zero-point contraction bodies: Zvt has no zero-point offset hardware, TOSA
    forbids non-zero `i16` zero points, and IREE expands `StableHLO`
    `!quant.uniform` (`StablehloLegalizeQuantToMath`) and named quantized Linalg
    ops (`QuantizedMatmulToMatmul`, `QuantizedConvToConv`) before codegen;
    inline zero-point `linalg.generic` bodies stay on RVV.

---

## 2. Proposed Change & Component Doc Delta

The architectural spec lives in [`matrix_codegen.md`](../matrix_codegen.md),
with boundary updates in
[`compiler/Transforms/CoralNPUTileSizeSelectionUtils.h`](../../../compiler/Transforms/CoralNPUTileSizeSelectionUtils.h)
and [`compiler/Transforms/Passes.td`](../../../compiler/Transforms/Passes.td).

### 16-Bit Zvt Microkernel & Drain Lowering

- Under Zvt v0.4 (`VLEN = 128`, tile edge `TE = 16`), signed `INT16 -> INT32`
  reuses `BF16 -> FP32`'s 16-bit operand layout (`SEW = 16, LMUL = 2, mtwiden =
  2`) alongside `INT8 -> INT32`'s signed `vtmms.tvv` (`vtype.altfmt = 1`) and
  32-bit row drain (`vtmv.v.t` + `vse32.v`) in
  `compiler/Transforms/CoralNPUMatrixCodegen.cpp` (with pass docs updated in
  `compiler/Transforms/Passes.td`; see `matrix_codegen.md` §2.3 for CSR values).
- Fused even-$K$ loops pack `TK = 2` steps per multiply (`204.8 GOPS` PE peak on
  $K$-contiguous operands; unpromoted strided `M x K` loads are LSU-bound) when
  every $K$-contiguous operand fills 16-element tiles, and use `TK = 1`
  (`102.4 GOPS` PE rate) otherwise, including unfused contractions. Unlike
  `BF16`, $M \le 2$ / `vecmat` has no dedicated 1D fastpath and falls back to
  RVV/host. `[1, 0]`-transposed scratch drains (`vlse32.v` + `vse32.v`) extend
  from `f32` to `i32` accumulators without asserting on `index` consumers.

### Contraction Legality, Scratch Budgeting & Roofline Scoring

- **Signed MAC & Producer Gating
  (`compiler/Transforms/CoralNPUTileSizeSelectionUtils.{h,cpp}`,
  `compiler/Transforms/CoralNPUTileSizeSelectionDTCM.cpp`,
  `compiler/Transforms/CoralNPUTileSizeSelectionFootprint.cpp`)**: Add `i16 x
  i16 -> i32` to `isZvtElementTypes` and require a pure FP or signed integer MAC
  body widened only by `arith.extf` or `arith.extsi`. Folding input readability
  into `isZvtMatrixContractionShape` rejects `arith.extui`, narrow multiplies,
  and zero-point bodies across DTCM tiling, RVV $K$-innermost selection, and
  roofline placement (dropping the duplicate guard in
  `CoralNPUTileSizeSelectionDTCM.cpp`). `getZvtLoadSource` requires a single
  1:1 single-use `arith.extf` / `arith.extsi` producer for Zvt root tiling,
  while DTCM scratch estimation relaxes use and map checks for RVV-tiled loops
  with cloned or transposed extensions.
- **Vector Lowering & Operand Guards
  (`compiler/Transforms/CoralNPUMatrixCodegen.cpp`,
  `iree-0006-llvmcpu-add-beforelowertollvmhook.patch`)**: Reject
  data-replicating `vector.broadcast` ops on contraction inputs (peeling scalar
  broadcasts in `BF16` `vecmat`), strip chained widening casts, and require
  memory-backed operand tiles on unfused contractions to avoid unbudgeted DTCM
  scratch. In
  `third_party/iree/compiler/src/iree/compiler/Codegen/LLVMCPU/LLVMCPUVirtualVectorLowering.cpp`
  (`iree-0006`), inspect both operands (including 1-channel convolution
  extracts, rejecting `arith.extui` and data-replicating `vector.broadcast`) and
  verify matching narrow input and 32-bit accumulator types before deferring
  contract lowering.
- **Stream Passthrough Clone Preservation
  (`iree-0016-stream-trace-tied-uses-through-passthrough.patch`)**: Step through
  variadic `util.optimization_barrier` and `arith.select` passthroughs in
  `third_party/iree/compiler/src/iree/compiler/Dialect/Stream/Analysis/ResourceUsage.cpp`
  and
  `third_party/iree/compiler/src/iree/compiler/Dialect/Stream/Transforms/ElideAsyncCopies.cpp`
  so protective Stream clones of constant, borrowed, or pre-passthrough
  resources (such as the seeded `linalg.mmt4d` accumulator in
  `tests/{spike,verilator}/matmul_i16_i32_16_8-8_16.mlir`) are preserved rather
  than elided.
- **LLVM Assembler Feature (`llvm-project-0001-add-zvt-support.patch`,
  `compiler/Target/CoralNPUTargetBackend.h`)**: Define `Zvti16i32mm` in
  `third_party/llvm-project/llvm/lib/Target/RISCV/RISCVFeatures.td`, accept
  `vtmms.tvv` and `vtmmu.tvv` under either `Zvti8i32mm` or `Zvti16i32mm` in
  `third_party/llvm-project/llvm/lib/Target/RISCV/RISCVInstrInfoZvt.td`, and
  include `+zvti16i32mm` in `CoralNPUTargetBackend`'s default feature string.
- **Roofline Placement (`compiler/Transforms/CoralNPUAffinityAnnotation.cpp`)**:
  Read pre-extension input bitwidths on Zvt roots via `getZvtLoadSource`,
  scoring full-tile `i16 -> i32` Zvt roots at `204.8 GOPS` (scaled by tile
  utilization on partial tiles) vs. `16.0 GOPS` host. Non-Zvt `i16 -> i32`
  roots with direct `i16` operands score at RVV's `12.8 GOPS` vs. `16.0 GOPS`
  host (`0.80x` peak, `~0.69x` at $32 \times 32$ with `800 ns` launch;
  `i32`-operand roots score `12.8 GOPS` RVV vs. `8.0 GOPS` host, and `i16 ->
  i16` scores `25.6 GOPS` RVV).

### MPACT, Spike & Verilator RTL Enablement

- **MPACT (`mpact-riscv-0004-zvt-i16-matmul.patch`, `MODULE.bazel`)**: Extend
  `vtmms.tvv` in `riscv/riscv_zvt_instructions.cc` (placed after `0001`/`0002`
  to cover both fast and generic loops) to accept `SEW = 16, TWIDEN = 2` with
  `uint32_t` modular accumulation matching RTL wraparound without signed UB.
  `MODULE.bazel` also advances `mpact-sim` (`9436377658` to `de9982fbec`) and
  `mpact-riscv` (`4262467a57` to `ef756043dc`, including out-of-range `vstart`
  CSR write clamping).
- **Spike (`build_tools/bazel/extensions.bzl`)**: Apply
  `@coralnpu_hw//third_party/spike:0010-Add-Zvti16i32mm-support.patch` to
  `riscv_isa_sim` (synced with `rules/repo_defs.bzl`,
  `third_party/spike.BUILD`, and `sw/coralnpu_sim/spike_simulator.h` in
  `third_party/coralnpu`) for `EXT_ZVTI16I32MM` (`SEW = 16, widen = 2`
  `vtmmu.tvv`/`vtmms.tvv` with `uint32_t` wraparound, 4-register tile alignment,
  and `SEW = 16` `altfmt = 1`).
- **Verilator RTL & FPGA (`third_party/coralnpu`)**: Bump `third_party/coralnpu`
  (`4d82d8701` to `eea8019a8`) so Verilator VME
  (`hdl/chisel/src/coralnpu/{Core,rvv/RvvCore}.scala`,
  `hdl/chisel/src/coralnpu/flags.bzl`) and Nexus FPGA matrix builds
  (`fpga/{BUILD,chip_nexus.core}`) enable `ZVTI16I32MM_ON`, and RVV/scalar
  decode (`hdl/chisel/src/coralnpu/{rvv/RvvDecode,scalar/Decode}.scala`,
  `hdl/verilog/rvv/design/RvvFrontEnd.sv`) accepts `SEW = 16, LMUL = 2,
  mtwiden = 2` integer `vtmm` ops with Slot-0 dynamic OP-VE traps (alongside
  Spike `0010`, TileLink-UL banked TCMs + `hw_sim` AXI fixes, RVV
  fault-only-first/interlock/mask-tail and float/LSU/ROB/branch decode fixes,
  scalar stall fixes on vector/matrix ops, opt-in `CoreAxi` transpose-DMA and
  memcopy bypass (`enableDma`, off by default), `CoreTlulCSR` multi-register
  writes, 16-bit Wide-SPI Nexus FPGA host support, and CoralFlow/host-CPU Bazel
  action bounds).

### Alternatives Considered

- **Gate compiler Zvt passes on `+zvti16i32mm`**: Deferred to §4. All four Zvt
  precisions currently gate pre-HAL placement, tiling, scratch budgeting, and
  microkernel codegen uniformly on `+zvtbase` (`matrix_codegen.md` §3), and
  Verilator VME and Nexus FPGA matrix builds enable `ZVTI16I32MM_ON` alongside
  `ZVT_ON`. Plumbing per-type sub-extension flags across those passes is tracked
  together for all Zvt sub-extensions in §4; until then, RTL builds with
  `+zvtbase` and `ZVTI16I32MM_ON` disabled trap at decode rather than falling
  back in the compiler.
- **Auto-promote non-widening `i16 -> i16` to `i32` (`i16 -> i32 -> i16`)**:
  Two's-complement `i32` accumulation plus narrowing `vncvt.x.x.w` truncation is
  bit-exact modulo $2^{16}$, but quantized ML models do not emit non-widening
  `i16 -> i16` contractions because a single `i16 * i16` product overflows
  `i16`. Rejected to avoid adding a promotion pass and narrowing drain without a
  motivating workload.

---

## 3. Implementation & Verification Plan

- [x] **Step 1 (`compiler/Transforms/CoralNPUTileSizeSelectionUtils.{h,cpp}`,
  `compiler/Transforms/CoralNPUTileSizeSelectionDTCM.cpp`,
  `compiler/Transforms/CoralNPUTileSizeSelectionFootprint.cpp`,
  `compiler/Transforms/CoralNPUAffinityAnnotation.cpp`)**: Unify signed `i16 ->
  i32` readability, scratch budgeting, and roofline tier scaling (§2).
- [x] **Step 2 (`compiler/Transforms/CoralNPUMatrixCodegen.cpp`,
  `compiler/Transforms/Passes.td`)**: Emit 16-bit `vtmms.tvv`, tighten
  broadcast/extension/unfused guards, add `i32` transpose drains & `Passes.td`.
- [x] **Step 3 (`iree-0006-llvmcpu-add-beforelowertollvmhook.patch`,
  `iree-0016-stream-trace-tied-uses-through-passthrough.patch`,
  `llvm-project-0001-add-zvt-support.patch`,
  `compiler/Target/CoralNPUTargetBackend.h`)**: Check both operands (with
  1-channel conv extracts, rejecting `arith.extui` and data-replicating
  broadcasts) and verify narrow input + 32-bit accumulator types in
  `isZvtMatrixContraction`, preserve constant, borrowed, or pre-passthrough
  Stream clones across `util.optimization_barrier` and `arith.select`, and
  enable `+zvti16i32mm` in LLVM and `CoralNPUTargetBackend`.
- [x] **Step 4 (`mpact-riscv-0004-zvt-i16-matmul.patch`, `MODULE.bazel`)**:
  Enable `SEW = 16, TWIDEN = 2` 16-bit integer lanes + `uint32_t` accumulation
  after `0002`, and bump `mpact-sim` and `mpact-riscv`.
- [x] **Step 5 (`build_tools/bazel/extensions.bzl`)**: Register
  `@coralnpu_hw//third_party/spike:0010-Add-Zvti16i32mm-support.patch` on
  `riscv_isa_sim` for `EXT_ZVTI16I32MM` (`SEW = 16, widen = 2` `vtmms.tvv` and
  `SEW = 16` `altfmt`).
- [x] **Step 6 (`third_party/coralnpu`)**: Bump submodule to `eea8019a8` for
  `ZVTI16I32MM_ON` Verilator VME / Nexus FPGA matrix support, Spike
  `0010-Add-Zvti16i32mm-support.patch`, and upstream RTL/cosim/DMA updates (§2).
- [x] **Step 7 (`examples/matmul-aot/`)**: Add `--int16` to `export_matmul.py`,
  `run_matmul.py`, and `test_matmul.sh` (MPACT and Verilator), plus
  `test_mmt4d.py`.
- [x] **Tests (`tests/transforms/`, `tests/spike/`, `tests/verilator/`)**:
  Positive/negative `CHECK`/`NOVSET`/`GATE`/`SMALL` tests in
  `matrix_codegen.mlir`, `zvt_i16_assembler.mlir`,
  `tile_size_selection_{dtcm,footprint}.mlir`, and `affinity_annotation.mlir`,
  IREE Stream lit tests in `iree-0016`, plus Spike, MPACT, and Verilator runs of
  `matmul_i16_i32_16_8-8_16.mlir`.
- [x] **Build, Doc & Archive Sync**: Register `zvt_i16_assembler.mlir` in
  `tests/transforms/CMakeLists.txt` and `matmul_i16_i32_16_8-8_16` in
  `tests/spike/{BUILD.bazel,CMakeLists.txt}` (`mpact` outside
  `if(CORALNPU_ENABLE_SPIKE)`) and
  `tests/verilator/{BUILD.bazel,CMakeLists.txt}`, update
  `docs/design/matrix_codegen.md`, archive to
  `docs/design/archive/zvt_int16.md`, and run `./scripts/format-code.sh`.

---

## 4. Follow-Up Items (For Reviewer Approval -> Buganizer component `2161972`)

- [ ] **Gate compiler Zvt passes on `+zvti16i32mm`** (`P3` |
  `compiler/Transforms/CoralNPUTileSizeSelectionUtils.{h,cpp}`,
  `compiler/Transforms/CoralNPUTileSizeSelectionFootprint.cpp`,
  `compiler/Transforms/CoralNPUAffinityAnnotation.cpp`,
  `compiler/Transforms/CoralNPUMatrixCodegen.cpp`,
  `iree-0006-llvmcpu-add-beforelowertollvmhook.patch`): Gate 16-bit integer Zvt
  tiling, scratch budgeting, placement, and codegen on `+zvti16i32mm` so RTL
  builds with `+zvtbase` and `ZVTI16I32MM_ON` disabled fall back to RVV in the
  compiler instead of trapping at decode.
