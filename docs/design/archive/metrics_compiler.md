# Add Per-Executable Compiler Metrics Bundle, Tile Grid, and Zvt Status

- **Author(s)**: Shaked Flur
- **Component Doc(s)**: [metrics.md](../metrics.md),
  [tiling_levels_explained.md](../tiling_levels_explained.md),
  [matrix_codegen.md](../matrix_codegen.md) *(updated in chained CL 2)*
- **Bug(s)**: None (searched component 2161972; no open issues found)

---

## 1. Summary & Motivation

Currently, `CoralNPUTargetBackend` (`compiler/Target/CoralNPUTargetBackend.cpp`)
can dump a register allocation report, and the Stream pipeline can dump an
affinity profile. However, no unified compile-time bundle reports target
configuration, linked ELF TCM usage, per-dispatch tile grids, or Zvt lowering
status and fallback reasons.

This change adds backend hooks (`CoralNPUTargetHooks`) and implements
`metrics/compiler/` to emit `coralnpu.metrics.v1` bundles and opt-in Zvt
missed-optimization remarks.

- **Goals**:
  - Decouple `metrics/compiler/` from `compiler/Target/` behind
    `CoralNPUTargetHooks` (`addConfigurationPasses`,
    `addPostMatrixCodegenPasses`, `onExecutableSerialized`) so only plugin
    registration (`compiler/Target/CoralNPUTarget.cpp`) depends on
    `metrics/compiler/`.
  - Emit per-executable `coralnpu.metrics.v1` JSON (`<executable>.metrics.json`)
    and `pretty` (`<executable>.metrics.txt`) bundles containing target
    configuration, per-dispatch metadata, and linked firmware ELF ITCM/DTCM
    region and section usage.
  - Record each dispatch root operation's effective tile sizes (`tile_grid` with
    `op`, `loops`, `workgroup`, `dtcm`, `vector`, and `zvt_grid`) and Zvt
    lowering status (`zvt = {candidate, lowered}`) via `coralnpu.metrics`
    (`RecordTileGridPass` and `RecordZvtLoweringPass`).
  - Emit opt-in `mlir::remark::missed` diagnostics (`remarkZvtMissed`, category
    `coralnpu-zvt`, subcategories `dtcm-tiling` and `matrix-codegen`) when a Zvt
    candidate stays on RVV during DTCM tiling or when no contraction in a
    dispatch function lowers to Zvt during matrix codegen.
- **Non-Goals**:
  - Aggregating the workload-level static Zvt lowering rate (`lowered /
    candidates` across bundles), which is implemented in the next batch of
    commits (`metrics/tools/report.py`).
  - Recording estimated DTCM tile byte footprint in `tile_grid` or tracking BF16
    `vecmat` K-loop lowering (`+zvfbfwma`; both tracked in **Section 4**).
  - Emitting ELF size reports for static library executables
    (`target.linkStatic`).

---

## 2. Proposed Change & Component Doc Delta

Section 2 of [`metrics.md`](../metrics.md),
[`tiling_levels_explained.md`](../tiling_levels_explained.md), and
[`matrix_codegen.md`](../matrix_codegen.md) document the post-change compiler
metrics pipeline and invariants:

- **Target Hook, Serialization Boundary & Zero-Codegen-Impact Invariant
  ([`metrics.md` §2.1](../metrics.md))**:
  Expose `CoralNPUTargetHooks` in `compiler/Target/CoralNPUTargetHooks.h`
  (`//compiler/Target:TargetHooks`) and `writeReport` / target-feature helpers
  in `compiler/Target/Utils.h` so `metrics/compiler/` attaches configuration and
  post-matrix-codegen passes and handles serialized executables without
  `CoralNPUTargetBackend` depending on `metrics/`. Enabling
  `--coralnpu-dump-metrics-format` only attaches discardable `coralnpu.metrics`
  function attributes and serializes the sideband report, never altering tile
  selection, pass pipelines, or generated code.
- **Linked Firmware ELF Accounting ([`metrics.md` §2.2](../metrics.md))**:
  For embedded dynamic libraries (`target.getLinkEmbedded()`), `parseMetricsElf`
  (`metrics/compiler/MetricsElf.h`) classifies allocated ELF sections against
  `__itcm_{start,size}` and `__dtcm_{start,size}` and computes `itcm.used` and
  `dtcm.used` from `PT_LOAD` segments (`p_memsz > 0`), subtracting overlaps with
  `.heap` and `.stack` sections in DTCM so `dtcm.used` matches runtime
  `firmware_memory.c` to the byte.
- **Per-Dispatch `coralnpu.metrics` Channel ([`metrics.md` §2.2](../metrics.md),
  [`tiling_levels_explained.md`](../tiling_levels_explained.md))**:
  `RecordTileGridPass` records
  `tile_grid = {op, loops, workgroup, dtcm, vector}` for static
  `TilingInterface` root ops (`getTiledRootOp`), clamping each inner level's
  positive tile sizes to its enclosing level's tile size and inheriting the
  enclosing size when a loop is left untiled (`0`). On `+zvtbase` targets,
  `RecordTileGridPass` also records `zvt = {candidate}` and, for any candidate
  (`isZvtMatrixContraction(rootOp)`), `tile_grid.zvt_grid = [ceil(vector[m]/16),
  ceil(vector[n]/16)]` on the dispatch function's discardable `coralnpu.metrics`
  `DictionaryAttr` (`metrics/compiler/MetricsAttr.h`). `RecordZvtLoweringPass`
  runs after `CoralNPUMatrixCodegenPass` to record `zvt.lowered` and prune
  `zvt_grid` when `lowered` is `false`.
- **Opt-In Zvt Missed Remarks ([`metrics.md` §2.2](../metrics.md),
  [`matrix_codegen.md` §3](../matrix_codegen.md))**:
  `CoralNPUTileSizeSelectionDTCM` (`dtcm-tiling`) and `CoralNPUMatrixCodegen`
  (`matrix-codegen`) call `remarkZvtMissed`
  (`compiler/Transforms/CoralNPUTileSizeSelectionUtils.h`) to explain via
  `--remarks-filter=coralnpu-zvt` why a Zvt candidate stayed on RVV during DTCM
  tiling or (at most once per function, on the first unlowered Zvt-typed
  contraction when no contraction in the function lowered to Zvt) why matrix
  codegen rejected it.

### Alternatives Considered

| Option                                                                 | Approach                                                                                                                                      | Pros                                                                                                                                                 | Cons                                                                                                          |
| :--------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------ |
| **Option 1 (Chosen)**<br/>*`[Lens: Modularity]` + `[Lens: Precedent]`* | Hook-based `metrics/compiler/` (`CoralNPUTargetHooks`) + discardable `coralnpu.metrics` `DictionaryAttr` on each dispatch + `remark::missed`. | Preserves one-way `metrics/ -> compiler/` dependency; threads per-dispatch facts across passes without shared mutable state; survives `-compile-to`. | Requires converting `coralnpu.metrics` `Attribute`s to JSON and adding hook points on `CoralNPUTargetBackend`. |
| **Option 2**<br/>*`[Lens: Minimalist]`*                                | Collect and dump metrics directly inside `CoralNPUTargetBackend` and `compiler/Transforms/` using an in-memory side table.                    | Fewer headers and no hook interface.                                                                                                                 | Couples `compiler/` to metrics schemas and requires mutex-guarded side tables across multi-threaded passes.   |
| **Option 3**<br/>*`[Lens: Shifter]`*                                   | Reconstruct all compiler metrics post-hoc from the emitted `.vmfb` ELF and MLIR remark files.                                                 | Zero changes to `CoralNPUTargetBackend` serialization or function attributes.                                                                        | Lowering configs (`tile_grid`) and unlowered Zvt candidates without remarks are lost once lowered to ELF.     |

**Decision Rationale**: Option 1 matches the `metrics/runtime/` hook boundary,
keeps `compiler/Target/` and `compiler/Transforms/` free of metrics formatting,
and uses MLIR's own IR (`coralnpu.metrics` on `FunctionOpInterface`) to carry
per-dispatch facts from configuration and codegen passes to serialization.

---

## 3. Implementation & Verification Plan

### CL 2a: Add Per-Executable Compiler Metrics Bundle and ELF Size Report

- [x] **Step 1 (`compiler/Target/*`, `compiler/Transforms/*`,
  `metrics/compiler/*`, `CMakeLists.txt`,
  `build_tools/cmake/iree_compiler_plugin.cmake`, `README.md`,
  `metrics/README.md`, `docs/design/*`)**: Add `CoralNPUTargetHooks`
  (`//compiler/Target:TargetHooks`), add `writeReport` and target-feature
  helpers in `compiler/Target/Utils.{h,cpp}`, implement `MetricsOptions`,
  `MetricsElf`, `MetricsBundle`, and `MetricsTargetHooks`, and wire
  `CoralNPUTarget.cpp` and CMake/Bazel targets.
- [x] **Tests (`metrics/tests/compiler/*`, `tests/BUILD.bazel`)**: *Positive*:
  Verify `json` and `pretty` bundles, `--coralnpu-dump-metrics-dir` (`-`, dir,
  empty), and ELF `itcm`/`dtcm` region and section summaries
  (`metrics_bundle.mlir`, `metrics_elf.mlir`; `bazel test --config=dev
  //metrics/tests/compiler:ci`). *Negative*: Verify default `none` emits no
  output and unknown `--coralnpu-dump-metrics-format=csv` fails fast.

### CL 2b: Add Per-Dispatch Tile Grid and `coralnpu.metrics` Attribute Channel

- [x] **Step 2 (`compiler/Target/*`, `compiler/Transforms/*`,
  `metrics/compiler/*`, `README.md`, `metrics/README.md`, `docs/design/*`)**:
  Add `CoralNPUTargetHooks::addConfigurationPasses`, implement `MetricsAttr`
  (`coralnpu.metrics`) and `RecordTileGridPass`
  (`tile_grid {op, loops, workgroup, dtcm, vector}`, clamping inner levels to
  their enclosing level's tile size), register compiler metrics passes via
  `registerMetricsPasses()`, and serialize `coralnpu.metrics` dictionary entries
  into each dispatch's bundle output.
- [x] **Tests (`metrics/tests/compiler/*`)**: *Positive*: Verify `tile_grid` on
  untiled and DTCM-tiled `linalg.matmul` and `linalg.generic` dispatches
  (`metrics_tile_grid.mlir`) and supported `coralnpu.metrics` attribute types
  (`metrics_bundle_attr.mlir`). *Negative*: Verify reserved dispatch keys,
  unsupported attribute types (`affine_map`), and dynamic-shape dispatches are
  skipped cleanly.

### CL 2c: Add Static Zvt Lowering Status and Missed-Optimization Remarks

- [x] **Step 3 (`compiler/Transforms/*`, `compiler/Target/*`,
  `metrics/compiler/*`)**: Add `remarkZvtMissed` and `getZvtMNLoops` in
  `CoralNPUTileSizeSelectionUtils.{h,cpp}` and `getZvtMatmulMnemonic` /
  `containsZvtMatmulMnemonic` in `compiler/Target/Utils.{h,cpp}`, emit
  `coralnpu-zvt` remarks in `CoralNPUTileSizeSelectionDTCM.cpp` (`dtcm-tiling`)
  and `CoralNPUMatrixCodegen.cpp` (`matrix-codegen`), add
  `addPostMatrixCodegenPasses`, extend `RecordTileGridPass` with `zvt.candidate`
  and `tile_grid.zvt_grid`, and implement `RecordZvtLoweringPass` (`zvt.lowered`
  and pruning `tile_grid.zvt_grid` when `lowered` is `false`).
- [x] **Tests & Archive Sync (`metrics/tests/compiler/*`, `README.md`,
  `metrics/README.md`, `docs/design/*`)**: *Positive*: Verify
  `zvt {candidate, lowered}`, `tile_grid.zvt_grid` on lowered candidates, and
  `--remarks-filter=coralnpu-zvt` YAML remarks (`metrics_zvt.mlir`). *Negative*:
  Verify non-lowered candidates ($K = 1$, small DTCM budget) omit `zvt_grid`
  when `lowered` is `false` and report missed remarks, and non-Zvt element types
  (`i32`) emit no Zvt remarks; sync component docs, `git mv` this doc to
  `docs/design/archive/metrics_compiler.md`, and run `./scripts/format-code.sh`.

### Implementation Notes & Adaptations

- **CL 2a (`metrics/compiler/MetricsElf.cpp`)**: Compute `itcm.used` and
  `dtcm.used` from `PT_LOAD` segments (`p_memsz > 0`), subtracting overlaps with
  `.heap` and `.stack` sections in DTCM so `dtcm.used` matches runtime
  `firmware_memory.c` to the byte.
- **CL 2c (`compiler/Transforms/CoralNPUTileSizeSelectionDTCM.cpp`)**: Emit
  `dtcm-tiling` missed remarks both when `!isZvtReadableInputs(root.op)` (fused
  non-widening input producer) and when `!setZvtVectorSizes(...)` (no Zvt tile
  fits the DTCM tile budget).

---

## 4. Follow-Up Items (For Reviewer Approval -> Buganizer `2161972`)

- [x] **Record DTCM tile footprint estimate in `RecordTileGridPass`** (`P2` |
      `metrics/compiler/RecordTileGridPass.cpp`,
      `compiler/Transforms/CoralNPUTileSizeSelectionFootprint.h`): Use
      `DTCMFootprintModel` to record estimated DTCM tile bytes in `tile_grid`.
- [x] **Unify standalone $M = 2$ Zvt contraction candidacy and tiling**
      (`P3` | `compiler/Transforms/CoralNPUTileSizeSelectionUtils.cpp`,
      `compiler/Transforms/CoralNPUTileSizeSelectionDTCM.cpp`): Evaluate
      recognizing and tiling standalone $M = 2$ contractions as Zvt candidates
      without regressing DTCM operand promotion or `subTileM` divisor snapping
      ($M = 1$ contractions are folded to 1D `vector.fma` by
      `DropVectorUnitDimsPass`).
- [x] **Track BF16 `vecmat` K-loop lowering and missed remarks (`+zvfbfwma`)**
      (`P3` | `metrics/compiler/*`,
      `compiler/Transforms/CoralNPUMatrixCodegen.cpp`): Record BF16 `vecmat`
      candidate and `vfwmaccbf16.vf` lowering status in `coralnpu.metrics` and
      emit missed remarks when falling back to generic RVV.
