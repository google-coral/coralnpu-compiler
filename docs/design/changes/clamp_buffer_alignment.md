# Clamp DTCM, Stack, and Constant Global Buffer Alignment to 16 Bytes

- **Author(s)**: sflur
- **Component Doc(s)**: [tiling_and_promotion.md](../tiling_and_promotion.md),
  [matrix_codegen.md](../matrix_codegen.md) *(updated in chained CL 2)*
- **Bug(s)**: N/A

---

## 1. Summary & Motivation

Today, MLIR One-Shot Bufferization stamps every promoted DTCM `memref.alloc`,
stack `memref.alloca`, and `.rodata` constant global (`memref.global` via
`IREEBufferizeConstantsPass`) with its 64-byte x86 cache-line default
(`BufferizationOptions::bufferAlignment = 64`). Similarly, MLIR's
`VectorTransferFullPartialRewriter` rewrite stamps temporary stack
`memref.alloca` buffers with 32-byte alignment. CoralNPU has no 64-byte cache
lines: current `VLEN = 128` hardware (`Sram_Nx128`-backed DTCM/ITCM,
`lsuDataBits = 128`, TileLink-UL crossbar, `VLEN = 128` RVV registers) and the
RISC-V stack ABI (`S128` = 128 bits) all operate on 16-byte beats. Aligning
buffers to 64 or 32 bytes wastes up to 48 bytes per buffer in
`DTCMFootprintModel`, `LLVMCPUAssignWorkgroupLocalMemoryPass`, and `.rodata` in
the 8 KB ITCM (turning a 16-byte `16xi8` bias slice into 64 bytes). On the 4 KB
stack (with a 3 KB static `alloca` cap), over-aligned `alloca` ops waste up to
48 bytes per buffer in `LLVMCPUCheckIRBeforeLLVMConversionPass` and force
`RISCVFrameLowering` to emit dynamic stack realignment (`andi sp, sp, -64` or
`-32`).

We lower CoralNPU's buffer alignment target to 16 bytes
(`kBufferAlignmentBytes`) across footprint estimation, Zvt scratch-tile
allocation, and a temporary clamping pass (`CoralNPUClampBufferAlignment`)
before local-memory packing, stack-size verification, and LLVM IR optimization.

- **Goals**:
  - Eliminate up to 48 bytes of 64-byte MLIR alignment padding per DTCM buffer
    and ITCM `.rodata` constant global in footprint estimation, packed
    `workgroup_local_memory`, and loop-indexed or $\le 16$-byte `.rodata`
    accesses (LLVM backend passes may still over-align offset-0 or
    constant-pooled `LMUL > 1` `.rodata` vectors to 32–128 bytes).
  - Eliminate dynamic stack realignment (`andi sp, sp, -64` / `-32`) and
    64-byte/32-byte stack-limit rounding on stack `memref.alloca` buffers while
    keeping 16-byte beat alignment.
- **Non-Goals**:
  - Patching `third_party/iree` or `third_party/llvm-project`: clamping is a
    temporary fix avoiding submodule patches until upstream options land (§4).
  - Changing 4 KB `local_memory` page rounding in IREE's `LibraryBuilder` and
    [`runtime/driver/coralnpu_command_buffer.c`](../../../runtime/driver/coralnpu_command_buffer.c)
    (part of the executable ABI), or 64-byte `local_memory` base alignment (at
    most 48 bytes once per dispatch) and per-binding 64-byte DDR tensor
    alignment (outside DTCM/ITCM) in
    [`runtime/sim/simulator_inline.c`](../../../runtime/sim/simulator_inline.c).

---

## 2. Proposed Change & Component Doc Delta

Consider a quantized `i8` matmul dispatch with a fused `16xi8` bias, an inlined
`.rodata` constant tensor, and a partial vector transfer spilling a temporary
to the stack. `DTCMFootprintModel` rounds each DTCM slice to 16 bytes instead
of 64 bytes, fitting larger tiles in the 24 KB DTCM budget. After upstream
bufferization emits 64-byte `memref.alloc` ops and vector splitting emits a
32-byte `memref.alloca`, `CoralNPUClampBufferAlignment` runs at the start of
`beforeLowerToLLVMHook` and clamps both to 16 bytes before CoralNPU hook passes
run. Downstream, `LLVMCPUAssignWorkgroupLocalMemoryPass` packs DTCM buffers on
16-byte boundaries, `LLVMCPUCheckIRBeforeLLVMConversionPass` checks the
3,072-byte `alloca` cap with 16-byte rounding, and `RISCVFrameLowering` emits a
standard 16-byte ABI prologue. After `buildLLVMCPUCodegenPassPipeline`, a
second clamping run lowers 64-byte `constant` `llvm.mlir.global` symbols from
`IREEBufferizeConstantsPass` to 16 bytes before LLVM IR optimization.

- **Shared 16-byte alignment constant (`kBufferAlignmentBytes`)**: Expose
  `kBufferAlignmentBytes` (16 bytes) in
  [`compiler/Target/Utils.h`](../../../compiler/Target/Utils.h) and remove the
  local 64-byte constant in
  [`compiler/Transforms/CoralNPUTileSizeSelectionFootprint.cpp`](../../../compiler/Transforms/CoralNPUTileSizeSelectionFootprint.cpp)
  (updating
  [`compiler/Transforms/CoralNPUTileSizeSelectionFootprint.h`](../../../compiler/Transforms/CoralNPUTileSizeSelectionFootprint.h)
  and [tiling_and_promotion.md](../tiling_and_promotion.md) §4.3 and §6.3). The
  constant targets current `VLEN = 128` hardware (`Sram_Nx128`,
  `lsuDataBits = 128`) and RISC-V `S128`; if CoralNPU adopts a 256-bit LSU
  (`Parameters.scala` defaults `lsuDataBits = 256`, splitting a 16-byte-aligned
  32-byte access across 2 beats), DTCM/ITCM alignment can be revisited while
  stack alignment stays bounded by `S128`.
- **Buffer and constant global clamping (`CoralNPUClampBufferAlignment`)**:
  Declare module-scoped `CoralNPUClampBufferAlignment`
  (`--coralnpu-clamp-buffer-alignment`) in
  [`compiler/Transforms/Passes.td`](../../../compiler/Transforms/Passes.td) and
  [`compiler/Transforms/Passes.h`](../../../compiler/Transforms/Passes.h), and
  schedule it in
  [`compiler/Target/CoralNPUTargetBackend.cpp`](../../../compiler/Target/CoralNPUTargetBackend.cpp)
  (see [tiling_and_promotion.md](../tiling_and_promotion.md) §3, §5.2, and §8.4,
  and [matrix_codegen.md](../matrix_codegen.md) §2.2):
  1. At the start of `beforeLowerToLLVMHook`, walking the module (including
     nested blocks before `HoistStaticallyBoundAllocationsPass`) to clamp
     upstream `memref.alloc`, `memref.alloca`, and `constant` `memref.global`
     ops without touching intentional $\ge 32$-byte allocations from later
     CoralNPU passes (such as 32-byte `workgroup_local` DMA descriptors).
  2. On the nested module pass manager right after
     `buildLLVMCPUCodegenPassPipeline` (skipping `llvm.func` bodies in
     pre-order walk) to clamp `constant` `llvm.mlir.global` ops lowered from
     `IREEBufferizeConstantsPass`.
- **Zvt DTCM scratch tiles (`CoralNPUMatrixCodegen`)**: Update `getScratchTile`
  in
  [`compiler/Transforms/CoralNPUMatrixCodegen.cpp`](../../../compiler/Transforms/CoralNPUMatrixCodegen.cpp)
  to allocate `#iree_codegen.workgroup_local` scratch tiles at
  `kBufferAlignmentBytes` instead of 64 bytes (see
  [tiling_and_promotion.md](../tiling_and_promotion.md) §3 and §7.3).

| Allocation / Global Case                                                   | Action          | Rationale                                                                      |
| :------------------------------------------------------------------------- | :-------------- | :----------------------------------------------------------------------------- |
| Explicit `alignment > 16`, scalar/`complex` elem (`constant` for globals)  | Clamp to `16` B | Matches 16 B DTCM/ITCM beat and `S128` stack ABI without over-padding.         |
| `alignment` unset, $\le 16$ (`1`–`16`), or non-`constant` global           | Leave unchanged | Preserves `1`–`8` B temporaries and mutable globals; DTCM `alloc`s floor 16 B. |
| `VectorType`-containing type, or direct `vector.type_cast` on alloc/alloca | Leave unchanged | Avoids `align 32`–`128` vector `LoadInst`/`StoreInst` on `align 16` data.      |

Because `CoralNPUClampBufferAlignment` is a temporary CoralNPU-pipeline pass,
it avoids transitive `ViewLikeOpInterface` walks, `SymbolTable::getSymbolUses`,
and `llvm.func` scans. Soundness rests on four pipeline invariants:
`splitVectorTransfersTo != "vector-transfers"` (`"linalg-copy"` or
`ConvTileAndDecomposeExpert`'s `"shuffle"`), `VectorTransferLoweringPass` uses
`enableFullUnroll()`, `ConvertToLLVMPass` runs
`populateVectorToLLVMConversionPatterns` with `useVectorAlignment = false`, and
`MemRefToLLVM` copies `memref.global` alignment only onto `llvm.mlir.global`.
Scalar loads/stores thus lower with ABI alignment $\le 16$ bytes, and vector
loads/stores on scalar buffers and `constant` globals emit scalar element
alignment ($\le 8$ bytes).

Downstream in LLVM, two backend passes can still over-align wide `.rodata`
vector constants (padding waste, not UB): `InferAlignmentPass` (in
`runLLVMIRPasses`) may re-inflate an internal `@__constant_...` global to
`getPrefTypeAlign` (32–128 bytes) on offset-0 `LMUL > 1` ($> 16$-byte) loads
(including iteration 0 of unrolled loops), and `SelectionDAG::getConstantPool`
(in `runEmitObjFilePasses`) aligns materialized constant-pool vectors to
`getPrefTypeAlign`. Loop-indexed and $\le 16$-byte constant accesses stay
16-byte aligned; `.rodata` alignment control is deferred to upstream LLVM (§4).

### Alternatives Considered

- **Clear `alignment` to `nullptr`**: Rejected. `MemRefToLLVM` lowers unset
  alignment to `align 0` (1–8 byte scalar ABI alignment), letting 16-byte RVV
  stack accesses straddle 16-byte DTCM SRAM row boundaries (2 beats vs. 1).
- **Patch `third_party/iree` and MLIR now**: Rejected. Fixing creation sites
  cleanly requires splitting buffer, stack, and vector alignment across both
  submodules; we defer that refactoring upstream (§4).
- **Transitive view / `SymbolTable::getSymbolUses` / `llvm.func` walk**:
  Rejected. The four pipeline invariants above already guarantee safety for
  this temporary pass without pointer/symbol-table tracing.
- **Per-symbol `.rodata.<sym>` sections on clamped `llvm.mlir.global` ops**:
  Rejected. Custom sections still cannot prevent `InstCombine` + `GlobalDCE` +
  `SelectionDAG::getConstantPool` from re-emitting folded vector constants at
  32–128 bytes, disable constant merging, and address only padding waste.
- **Fold into `CoralNPULowerDTCMCopies` / `CoralNPUMatrixCodegen`, or move
  `IREEBufferizeConstantsPass` earlier**: Rejected. Both passes have unrelated
  scopes, and running `IREEBufferizeConstantsPass` before
  `ConvertUnsupportedFloatToIntBuffersPass` breaks `bf16` constant globals.

---

## 3. Implementation & Verification Plan

- [ ] **Step 1 (`compiler/Target/Utils.h`,
  `compiler/Transforms/CoralNPUTileSizeSelectionFootprint.{h,cpp}`)**: Define
  `kBufferAlignmentBytes` (16 bytes) and remove the local 64-byte constant.
- [ ] **Step 2 (`compiler/Transforms/Passes.{td,h}`,
  `compiler/Transforms/CoralNPUClampBufferAlignment.cpp`,
  `compiler/Target/CoralNPUTargetBackend.cpp`)**: Implement module-scoped
  `CoralNPUClampBufferAlignmentPass` per the §2 clamping table and schedule it
  at the start of `beforeLowerToLLVMHook` and after
  `buildLLVMCPUCodegenPassPipeline`.
- [ ] **Step 3 (`compiler/Transforms/CoralNPUMatrixCodegen.cpp`)**: Update
  `getScratchTile` to use `kBufferAlignmentBytes` (16 bytes) instead of 64.
- [ ] **Tests (`tests/transforms/clamp_buffer_alignment.mlir`,
  `tests/transforms/matrix_codegen.mlir`,
  `tests/transforms/tile_size_selection_{dtcm,footprint}.mlir`,
  `tests/transforms/promote_operands_pipeline.mlir`)**:
  - *Positive*: Test 64 B / 32 B clamping (including nested `alloca`s and
    `constant` globals) in `clamp_buffer_alignment.mlir`; update the 6
    `alignment = 64` checks in `matrix_codegen.mlir`; add 16-byte-multiple and
    constant-global checks in `promote_operands_pipeline.mlir`; add a 16 B vs.
    64 B tile-selection case in `tile_size_selection_dtcm.mlir` and update
    comments in `tile_size_selection_{dtcm,footprint}.mlir`; run
    `bazel test --config=dev //tests:ci`.
  - *Negative / Fallback*: Verify every `Leave unchanged` case from the §2
    table (including unset alignment on all four ops and non-`constant`
    `memref.global` / `llvm.mlir.global` ops) in `clamp_buffer_alignment.mlir`.
- [ ] **Build, Doc & Archive Sync**: Sync `compiler/Transforms/BUILD.bazel`,
  `compiler/Transforms/CMakeLists.txt`, and `tests/transforms/CMakeLists.txt`,
  verify [tiling_and_promotion.md](../tiling_and_promotion.md),
  [matrix_codegen.md](../matrix_codegen.md), and [INDEX.md](../INDEX.md),
  `git mv` this doc to `docs/design/archive/`, and run
  `./scripts/format-code.sh`.

### Implementation Notes & Adaptations

- *(None yet)*

---

## 4. Follow-Up Items (`<component>.md` TODOs / Buganizer `2161972`)

- [ ] **Upstream separated buffer, stack, and vector alignment controls across
  IREE, MLIR, and LLVM** (`tiling_and_promotion.md` |
  `compiler/Transforms/CoralNPUClampBufferAlignment.cpp`): Split the three
  conflated alignment concerns upstream so 16-byte alignment is emitted at
  creation time and `CoralNPUClampBufferAlignment` can be removed:
  - *Memory / cache-line (`bufferAlignment`)*: Target-configurable option in
    IREE's `CPUCodegenOptions`, `IREEComprehensiveBufferizePass`, and
    `IREEBufferizeConstantsPass` for DTCM `alloc` and `.rodata` `memref.global`.
  - *Stack (`stackAlignment`)*: ABI-bounded (`S128`) option in `cpuAllocationFn`
    (skipping `VectorType` elements) and MLIR's
    `VectorTransferFullPartialRewriter` (`LinalgCopy` mode only, replacing its
    32-byte default without weakening `VectorTransfer` mode).
  - *Natural vector ABI / preferred alignment*: Keep `DataLayout` vector ABI
    separate for `VectorType` allocations, and address LLVM `.rodata` vector
    re-inflation (`InferAlignmentPass`, `SelectionDAG::getConstantPool`) via
    LLVM `DataLayout` / `TTI`.
